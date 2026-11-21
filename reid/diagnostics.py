"""Resumo das tracks antes e depois da reidentificação."""

from collections import defaultdict

import numpy as np


def _label(group):
    category, team, role = group
    if role == "referee":
        return "Árbitros"
    name = "sem equipa" if team is None else f"Equipa {team}"
    return f"{name} (guarda-redes)" if role == "goalkeeper" else name


def summarize(reidentifier):
    """Devolve linhas de texto, por grupo: tracks, identidades, ligações."""
    mapping = reidentifier.global_ids()
    groups = defaultdict(list)
    for t in reidentifier.tracklets.values():
        groups[t.group].append(t)

    lines = []
    for group in sorted(groups, key=lambda g: (g[0], str(g[1]), g[2])):
        members = groups[group]
        long_enough = [t for t in members if t.length >= reidentifier.min_length]
        identities = {mapping[t.track_id] for t in members}
        links = sum(1 for t in members if t.track_id in reidentifier.links)
        lengths = [t.length for t in members]
        lines.append(
            f"{_label(group):<26} {len(members):>3} tracks "
            f"({len(members) - len(long_enough)} curtas) -> "
            f"{len(identities):>3} identidades | {links:>3} ligações | "
            f"duração mediana {int(np.median(lengths))} frames"
        )
    lines.append(f"Ligações rejeitadas por ambiguidade: {reidentifier.rejected}")
    return lines
