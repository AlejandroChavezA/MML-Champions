#!/usr/bin/env python3
"""
Paso 0 (MVP): CLI que imprime EXACTAMENTE los formatos del contrato.

Este script NO hace predicción real todavía. Solo valida el formato de salida.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

try:
    from prediction.output_contract import (
        MatchPredictionRow,
        DetailedMatchPrediction,
        DetailedSignals,
        render_prediccion_jornada_completa,
        render_prediccion_detallada_match,
    )
except ModuleNotFoundError:
    # Permite ejecutar como script: `python3 prediction/cli_step0.py`
    from pathlib import Path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from prediction.output_contract import (  # type: ignore
        MatchPredictionRow,
        DetailedMatchPrediction,
        DetailedSignals,
        render_prediccion_jornada_completa,
        render_prediccion_detallada_match,
    )


@dataclass(frozen=True)
class _Fixture:
    date_iso: str
    matchday: int
    home_team: str
    away_team: str


def _load_champions_fixtures(season: str = "2025-26") -> list[_Fixture]:
    path = Path("data/cleaned/champions") / f"matches_{season.replace('-', '_')}.csv"
    if not path.exists():
        return []
    fixtures: list[_Fixture] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                fixtures.append(
                    _Fixture(
                        date_iso=row["date"],
                        matchday=int(float(row["matchday"])),
                        home_team=row["home_team"],
                        away_team=row["away_team"],
                    )
                )
            except Exception:
                continue
    return fixtures


def _matchdays(fixtures: list[_Fixture]) -> list[int]:
    return sorted({f.matchday for f in fixtures})


def _probs_for_pair(home: str, away: str) -> tuple[float, float, float]:
    # determinístico para el demo (no es modelo real)
    h = hashlib.sha256(f"{home}__{away}".encode("utf-8")).digest()
    a = (h[0] + 1) / 256.0
    b = (h[1] + 1) / 256.0
    c = (h[2] + 1) / 256.0
    s = a + b + c
    return a / s, b / s, c / s  # home, draw, away


def _label_from_probs(ph: float, pd: float, pa: float) -> tuple[str, float]:
    mx = max(ph, pd, pa)
    if mx == ph:
        return "LOCAL", mx * 100
    if mx == pa:
        return "VISITANTE", mx * 100
    return "EMPATE", mx * 100


def _abbr(team: str) -> str:
    # código corto estilo "WOL" / "AVL"
    cleaned = "".join(ch for ch in team.upper() if ch.isalnum() or ch.isspace())
    parts = [p for p in cleaned.split() if p not in {"FC", "CF", "SC", "FK", "AC"}]
    if not parts:
        parts = cleaned.split()
    letters = "".join(p[0] for p in parts if p)[:3]
    if len(letters) < 3:
        letters = (letters + (parts[0] if parts else "XXX"))[:3]
    return letters


_WEEKDAYS_ES = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]


def _fmt_fecha_linea(date_iso: str) -> str:
    # date_iso: 2025-09-16T18:45:00Z
    dt = datetime.fromisoformat(date_iso.replace("Z", "+00:00")).astimezone(timezone.utc)
    wd = _WEEKDAYS_ES[dt.weekday()]
    hour = dt.hour
    minute = dt.minute
    ampm = "AM" if hour < 12 else "PM"
    hour12 = hour % 12
    if hour12 == 0:
        hour12 = 12
    return f"📅 {wd} {hour12}:{minute:02d} {ampm}"


def demo_jornada_completa() -> tuple[int, list[_Fixture], int, int]:
    fixtures = _load_champions_fixtures("2025-26")
    mds = _matchdays(fixtures)
    if not mds:
        # fallback: mantiene el contrato aunque no haya datos
        jornada = 1
        print(
            render_prediccion_jornada_completa(
                jornadas_min=1,
                jornadas_max=1,
                jornada=jornada,
                rows=[],
            )
        )
        return jornada, [], 1, 1

    jornadas_min, jornadas_max = mds[0], mds[-1]
    jornada = jornadas_min
    jornada_fixtures = [f for f in fixtures if f.matchday == jornada]

    rows: list[MatchPredictionRow] = []
    for f in jornada_fixtures[:10]:
        ph, pd, pa = _probs_for_pair(f.home_team, f.away_team)
        label, conf = _label_from_probs(ph, pd, pa)
        rows.append(MatchPredictionRow(f.home_team, f.away_team, label, conf))

    print(
        render_prediccion_jornada_completa(
            jornadas_min=jornadas_min,
            jornadas_max=jornadas_max,
            jornada=jornada,
            rows=rows,
        )
    )
    return jornada, fixtures, jornadas_min, jornadas_max


def demo_detallada(*, jornada: int, fixtures: list[_Fixture], jornadas_min: int, jornadas_max: int) -> None:
    pick = next((f for f in fixtures if f.matchday == jornada), None)
    if not pick:
        return

    ph, pd, pa = _probs_for_pair(pick.home_team, pick.away_team)
    label, conf = _label_from_probs(ph, pd, pa)
    home_abbr = _abbr(pick.home_team)
    away_abbr = _abbr(pick.away_team)

    if label == "LOCAL":
        modelo_dice = f"🏠 GANA {home_abbr}"
    elif label == "VISITANTE":
        modelo_dice = f"✈️ GANA {away_abbr}"
    else:
        modelo_dice = "🤝 EMPATE"

    breakdown = f"{home_abbr}: {ph*100:.1f}% chance | {away_abbr}: {pa*100:.1f}% chance | Empate: {pd*100:.1f}% chance"

    m = DetailedMatchPrediction(
        partido=f"{home_abbr} @ {away_abbr}",
        fecha_linea=_fmt_fecha_linea(pick.date_iso),
        modelo_dice=modelo_dice,
        confianza_pct=conf,
        breakdown_line=breakdown,
        pro_title=f"✅ ¿POR QUÉ FAVORECE A {away_abbr if label == 'VISITANTE' else home_abbr}?",
        con_title=f"❌ ¿QUÉ FAVORECE A {home_abbr if label == 'VISITANTE' else away_abbr}?",
        signals=DetailedSignals(
            pro=[
                "Señal del modelo: (demo) Diferencia margen últimos 20 partidos (+0.000) ⭐",
                "Señal del modelo: (demo) Forma últimos 5 partidos (+0.000) ⭐",
                "Señal del modelo: (demo) Goles promedio últimos 5 (+0.000) ⭐",
                "Señal del modelo: (demo) Ventaja casa/visitante (+0.000) ⭐",
            ],
            con=[
                "Señal del modelo: (demo) Diferencia margen últimos 20 partidos (--0.000) ⭐",
                "Señal del modelo: (demo) Forma últimos 5 partidos (--0.000) ⭐",
                "Señal del modelo: (demo) Goles promedio últimos 5 (--0.000) ⭐",
                "Señal del modelo: (demo) Ventaja casa/visitante (--0.000) ⭐",
            ],
        ),
    )
    print("")
    print(" -Formato para prediccion jornada completa detallada(formato unicamente):")
    print(" PREDICCIÓN POR JORNADA (DETALLADA)")
    print("============================================================")
    print(f"Jornadas disponibles: {jornadas_min} - {jornadas_max}")
    print(f"Jornadas completadas: {jornadas_max}")
    print(f"Próxima jornada: {jornada}")
    print("")
    print("Opciones:")
    print("1. Seleccionar jornada específica")
    print("2. Siguiente jornada no completada")
    print("3. Ver última jornada completada")
    print("Selecciona opción (1-3): 2")
    print("")
    print(f"Analizando jornada {jornada}...")
    print("")
    print(render_prediccion_detallada_match(m=m))
    print("────────────────────────────────────────────────────────────────────────────────")
    print("Presiona Enter para continuar...")


def main() -> None:
    jornada, fixtures, jornadas_min, jornadas_max = demo_jornada_completa()
    demo_detallada(
        jornada=jornada,
        fixtures=fixtures,
        jornadas_min=jornadas_min,
        jornadas_max=jornadas_max,
    )


if __name__ == "__main__":
    main()

