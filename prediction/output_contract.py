"""
Contrato de salida (FORMATO ÚNICAMENTE).

Este módulo define funciones que construyen exactamente los formatos mostrados en:
`prediction/MVP_PLAN_prediction.txt`

Regla: NO cambiar saltos, separadores ni emojis sin actualizar el contrato.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class MatchPredictionRow:
    local: str
    visitante: str
    prediccion: str  # "LOCAL" | "EMPATE" | "VISITANTE"
    confianza_pct: float  # 0-100
    real: str = "N/A"


def _fmt_pct(x: float) -> str:
    return f"{x:.1f}%"


def render_prediccion_jornada_completa(
    *,
    jornadas_min: int,
    jornadas_max: int,
    jornada: int,
    rows: Iterable[MatchPredictionRow],
) -> str:
    """
    Render del bloque:
    - PREDICCIÓN DE JORNADA COMPLETA
    - PREDICCIONES JORNADA {jornada}
    - tabla
    - "Presiona Enter para continuar..."
    """
    lines: List[str] = []
    lines.append("PREDICCIÓN DE JORNADA COMPLETA")
    lines.append("==================================================")
    lines.append(f"Jornadas disponibles: {jornadas_min} - {jornadas_max}")
    lines.append(f"Ingresa el número de jornada: {jornada}")
    lines.append("")
    lines.append(f" Prediciendo jornada {jornada}...")
    lines.append("")
    lines.append(f" PREDICCIONES JORNADA {jornada}")
    lines.append("================================================================================")
    lines.append("Local                     Visitante                 Predicción   Confianza  Real    ")
    lines.append("--------------------------------------------------------------------------------")

    # Anchors para que el layout sea estable (como el ejemplo)
    w_local = 26
    w_visit = 26
    w_pred = 11
    w_conf = 10
    w_real = 7

    for r in rows:
        local = (r.local or "")[:w_local].ljust(w_local)
        visit = (r.visitante or "")[:w_visit].ljust(w_visit)
        pred = (r.prediccion or "")[:w_pred].ljust(w_pred)
        conf = _fmt_pct(r.confianza_pct).rjust(w_conf)
        real = (r.real or "N/A")[:w_real].ljust(w_real)
        lines.append(f"{local} {visit} {pred} {conf}      {real}")

    lines.append("Presiona Enter para continuar...")
    return "\n".join(lines)


@dataclass(frozen=True)
class DetailedSignals:
    pro: List[str]
    con: List[str]


@dataclass(frozen=True)
class DetailedMatchPrediction:
    partido: str  # ej: "WOL @ AVL"
    fecha_linea: str  # ej: "📅 Viernes 8:00 PM"
    modelo_dice: str  # ej: "✈️ GANA AVL" o "🏠 GANA LIV"
    confianza_pct: float
    breakdown_line: str  # ej: "WOL: 17.8% chance | AVL: 64.6% chance | Empate: 17.6% chance"
    pro_title: str = "✅ ¿POR QUÉ FAVORECE A AVL?"
    con_title: str = "❌ ¿QUÉ FAVORECE A WOL?"
    signals: Optional[DetailedSignals] = None


def render_prediccion_detallada_match(*, m: DetailedMatchPrediction) -> str:
    """
    Render del bloque de partido detallado (formato exactamente como el ejemplo).
    """
    lines: List[str] = []
    lines.append("────────────────────────────────────────────────────────────────────────────────")
    lines.append(f"⚽ PARTIDO: {m.partido}")
    lines.append(m.fecha_linea)
    lines.append("────────────────────────────────────────────────────────────────────────────────")
    lines.append("")
    lines.append(f"🎯 EL MODELO DICE: {m.modelo_dice}")
    lines.append(f"   Confianza: {_fmt_pct(m.confianza_pct)}")
    lines.append(f"   {m.breakdown_line}")
    lines.append("")
    lines.append(m.pro_title)
    lines.append("────────────────────────────────────────────────────────────────────────────────")
    if m.signals and m.signals.pro:
        for i, s in enumerate(m.signals.pro, 1):
            lines.append(f"  {i}. {s}")
    else:
        lines.append("  1. Señal del modelo: N/A")
    lines.append("")
    lines.append(m.con_title)
    lines.append("────────────────────────────────────────────────────────────────────────────────")
    if m.signals and m.signals.con:
        for i, s in enumerate(m.signals.con, 1):
            lines.append(f"  {i}. {s}")
    else:
        lines.append("  1. Señal del modelo: N/A")
    lines.append("")
    return "\n".join(lines)

