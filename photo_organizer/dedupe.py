"""Stage 2a: find exact duplicate copies and choose a keeper (README: Duplicate copies).

Pure logic over manifest rows + metadata; the caller writes the outcome to the
manifest. Duplicates are only *marked* here, never moved or modified.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from .metadata import Decision, FileMeta, keys_agree

PICTURE_KINDS = ("nef", "jpg", "heic")
PARTNER_KIND = {"nef": "jpg", "jpg": "nef"}


@dataclass(frozen=True)
class PicFile:
    path: str
    kind: str
    size: int
    mtime_ns: int
    image_hash: str | None
    sha256: str | None
    meta: FileMeta | None
    decision: Decision
    folder: str          # folder hint

    @property
    def stem(self) -> str:
        return PurePosixPath(self.path).name.rsplit(".", 1)[0].lower()

    @property
    def directory(self) -> str:
        return str(PurePosixPath(self.path).parent)

    def capture_key(self) -> tuple:
        return self.meta.capture_key() if self.meta else (None, None, None, None)


@dataclass
class DupGroup:
    keeper: PicFile
    copies: list[PicFile]              # marked duplicate_copy
    reason: str                        # which keeper rule decided


@dataclass
class Conflict:
    files: list[PicFile]               # copies with different human decisions; none marked


@dataclass
class DedupeResult:
    groups: list[DupGroup] = field(default_factory=list)
    conflicts: list[Conflict] = field(default_factory=list)
    # duplicate path -> what it has that the keeper lacks
    extra_metadata: dict[str, str] = field(default_factory=dict)
    # same pixels but different capture data (not treated as copies)
    odd_matches: list[list[str]] = field(default_factory=list)

    def role_of(self) -> dict[str, tuple[str, str | None]]:
        """path -> (role, duplicate_of) for every file involved in a copy group."""
        roles: dict[str, tuple[str, str | None]] = {}
        for g in self.groups:
            roles[g.keeper.path] = ("keeper", None)
            for c in g.copies:
                roles[c.path] = ("duplicate_copy", g.keeper.path)
        for c in self.conflicts:
            first = c.files[0]
            roles[first.path] = ("keeper", None)
            for f in c.files[1:]:
                roles[f.path] = ("conflict", first.path)
        return roles


def _has_partner_next_to_it(f: PicFile, by_stem: dict[str, list[PicFile]]) -> bool:
    want = PARTNER_KIND.get(f.kind)
    if want is None:
        return False
    return any(g.kind == want and g.folder.lower() == f.folder.lower()
               and f.capture_key()[0] is not None
               and keys_agree(f.capture_key(), g.capture_key())
               for g in by_stem.get(f.stem, []))


def _pref_rank(path: str, preferred_roots: tuple[str, ...]) -> int:
    p = path.lower()
    for i, root in enumerate(preferred_roots):
        r = root.strip("/").lower()
        if p == r or p.startswith(r + "/"):
            return i
    return len(preferred_roots)


def find_duplicates(pics: list[PicFile], preferred_roots: tuple[str, ...] = ()) -> DedupeResult:
    result = DedupeResult()

    by_stem: dict[str, list[PicFile]] = defaultdict(list)
    for f in pics:
        by_stem[f.stem].append(f)

    # "Fuller folder": distinct captures per folder hint.
    folder_caps: dict[str, set] = defaultdict(set)
    for f in pics:
        key = f.capture_key()
        folder_caps[f.folder.lower()].add(key if key[0] is not None else f.path)

    by_hash: dict[tuple[str, str], list[PicFile]] = defaultdict(list)
    for f in pics:
        if f.image_hash:
            by_hash[(f.kind, f.image_hash)].append(f)

    for (_kind, _h), same in sorted(by_hash.items()):
        if len(same) < 2:
            continue
        # Capture metadata must match too; split by capture key.
        by_key: dict[tuple, list[PicFile]] = defaultdict(list)
        for f in same:
            by_key[f.capture_key()].append(f)
        if len(by_key) > 1:
            result.odd_matches.append(sorted(f.path for f in same))
        for copies in by_key.values():
            if len(copies) >= 2:
                _resolve(sorted(copies, key=lambda f: f.path), by_stem, folder_caps,
                         preferred_roots, result)
    result.groups.sort(key=lambda g: g.keeper.path)
    result.conflicts.sort(key=lambda c: c.files[0].path)
    return result


def _resolve(copies, by_stem, folder_caps, preferred_roots, result: DedupeResult) -> None:
    # Rule 1: human decisions.
    decided = {f.decision for f in copies if f.decision}
    if len(decided) > 1:
        result.conflicts.append(Conflict(_ranked(copies, by_stem, folder_caps, preferred_roots)))
        return
    candidates = [f for f in copies if f.decision] if decided else copies
    reason = "human decision" if decided and len(candidates) < len(copies) else None

    ranked = _ranked(candidates, by_stem, folder_caps, preferred_roots)
    keeper = ranked[0]
    if reason is None:
        reason = _deciding_rule(ranked, by_stem, folder_caps, preferred_roots)
    dups = [f for f in copies if f is not keeper]
    result.groups.append(DupGroup(keeper, dups, reason))

    for d in dups:
        extra = []
        if d.decision.keywords - keeper.decision.keywords:
            extra.append("keywords " + ",".join(sorted(d.decision.keywords - keeper.decision.keywords)))
        if d.meta and keeper.meta and d.meta.has_gps and not keeper.meta.has_gps:
            extra.append("GPS")
        if d.decision.rating is not None and keeper.decision.rating is None:
            extra.append(f"rating {d.decision.rating}")
        if extra:
            result.extra_metadata[d.path] = "; ".join(extra)


def _rank_key(f: PicFile, by_stem, folder_caps, preferred_roots) -> tuple:
    return (
        not _has_partner_next_to_it(f, by_stem),        # rule 2: paired beats lone
        -len(folder_caps[f.folder.lower()]),             # rule 3: fuller folder
        _pref_rank(f.path, preferred_roots),             # rule 4: preferred roots
        f.mtime_ns, len(f.path), f.path,                 # rule 5: oldest, shortest, stable
    )


def _ranked(files, by_stem, folder_caps, preferred_roots) -> list[PicFile]:
    return sorted(files, key=lambda f: _rank_key(f, by_stem, folder_caps, preferred_roots))


def _deciding_rule(ranked, by_stem, folder_caps, preferred_roots) -> str:
    names = ["paired beats lone", "fuller folder", "preferred root",
             "oldest file", "shortest path", "path order"]
    a = _rank_key(ranked[0], by_stem, folder_caps, preferred_roots)
    b = _rank_key(ranked[1], by_stem, folder_caps, preferred_roots)
    for name, x, y in zip(names, a, b):
        if x != y:
            return name
    return "path order"


def folder_report(pics: list[PicFile], roles: dict[str, tuple[str, str | None]],
                  folder_of: dict[str, str]) -> list[tuple[str, int, int, list[tuple[str, int]]]]:
    """Folders containing duplicate copies: (folder, dup count, total, [(keeper folder, n)]).

    dup count == total means a duplicate folder (100%)."""
    totals = Counter(f.folder for f in pics)
    dups: dict[str, Counter] = defaultdict(Counter)
    for f in pics:
        role, of = roles.get(f.path, ("keeper", None))
        if role == "duplicate_copy" and of:
            dups[f.folder][folder_of[of]] += 1
    out = []
    for folder, keeper_folders in sorted(dups.items()):
        out.append((folder, sum(keeper_folders.values()), totals[folder],
                    keeper_folders.most_common()))
    return out
