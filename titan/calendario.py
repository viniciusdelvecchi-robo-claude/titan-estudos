"""Calendário de pregões por ativo e por data do servidor (F1-13, emenda C5). Autor: Claude.

Fuso do servidor: medido nas barras (medidas/calendario_inferido.json) — o câmbio abre segunda 00:00 do
servidor o ano todo, então servidor = Nova York + 7 h (GMT+2 no inverno dos EUA, GMT+3 no verão).
A confirmar no terminal ao vivo (oráculo, F1-9).

O calendário histórico é inferido da presença das barras e só aceita dia que bate com o padrão do regime
(revisão do Codex, achados 5 e 6): o que não se comprova fica fora, isto é, inapto. A hora limite de
zeragem fica numa fronteira de barra (emenda C7): fechamento menos MARGEM_ZERAGEM.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import UTC, date, datetime, time, timedelta
from typing import Iterable, Mapping
from zoneinfo import ZoneInfo

from .tipos import Sessao

NOVA_YORK = ZoneInfo("America/New_York")
SERVIDOR_MENOS_NY = timedelta(hours=7)
BARRA = timedelta(minutes=15)
MARGEM_ZERAGEM = timedelta(minutes=15)       # convenção: zera no início da última barra do pregão
MINIMO_CONCORDANTES = 3                      # convenção: sessões iguais para um regime ter padrão
FONTE_INFERIDA = "inferido da presença das barras de 15 minutos; servidor = Nova York + 7 h"


def offset_servidor_min(instante_utc: datetime) -> int:
    """Minutos que o relógio do servidor está à frente do UTC naquele instante."""
    if instante_utc.tzinfo is None:
        raise ValueError("instante precisa de fuso")
    ny = instante_utc.astimezone(NOVA_YORK).utcoffset()
    return int((ny + SERVIDOR_MENOS_NY).total_seconds() // 60)


def servidor_para_utc(relogio_servidor: datetime) -> datetime:
    """Converte hora do servidor (sem fuso, como o MT5 grava) para UTC consciente.

    Na mudança de horário a hora local pode ser ambígua; usa-se o offset de Nova York daquela hora
    local (fold=0), o mesmo que o servidor usa por construção.
    """
    if relogio_servidor.tzinfo is not None:
        raise ValueError("hora do servidor vem sem fuso")
    ny_local = (relogio_servidor - SERVIDOR_MENOS_NY).replace(tzinfo=NOVA_YORK)
    return ny_local.astimezone(UTC)


def epoca_servidor_para_utc(epoca: int) -> datetime:
    """O campo time do MT5 é a hora do servidor escrita como se fosse UTC."""
    return servidor_para_utc(datetime.fromtimestamp(epoca, UTC).replace(tzinfo=None))


def meia_noite_servidor_utc(dia: date) -> datetime:
    return servidor_para_utc(datetime.combine(dia, time()))


def horario_verao_europa(dia: date) -> bool:
    """Verão da União Europeia: do último domingo de março ao último domingo de outubro."""
    def ultimo_domingo(m):
        d = date(dia.year, m, 31)
        return d - timedelta(days=(d.weekday() + 1) % 7)
    return ultimo_domingo(3) <= dia < ultimo_domingo(10)


def horario_verao_eua(dia: date) -> bool:
    return offset_servidor_min(datetime.combine(dia, time(12), tzinfo=UTC)) == 180


def construir_calendario(simbolo: str, epocas_servidor: Iterable[int]) -> tuple[dict[date, Sessao], dict[date, str]]:
    """Calendário comprovável e dias excluídos (com motivo).

    Uma sessão só entra se o dia tem barras e se abertura e fechamento coincidem com o padrão do mesmo
    símbolo no mesmo regime (dia da semana, verão dos EUA, verão da Europa) e não há buraco dentro do pregão.
    Dia útil sem barra, extremidade fora do padrão ou buraco: excluído, logo inapto (ausência não prova
    feriado; extremidade não confirmada pode ser falha de coleta).
    """
    por_dia: dict[date, list[datetime]] = defaultdict(list)
    for e in epocas_servidor:
        local = datetime.fromtimestamp(int(e), UTC).replace(tzinfo=None)
        por_dia[local.date()].append(local)
    if not por_dia:
        return {}, {}
    regime = lambda d: (d.weekday(), horario_verao_eua(d), horario_verao_europa(d))
    extremos = {d: (min(b).time(), (max(b) + BARRA).time()) for d, b in por_dia.items() if d.weekday() < 5}
    grupos: dict[tuple, list[tuple]] = defaultdict(list)
    for d, ex in extremos.items():
        grupos[regime(d)].append(ex)
    # padrão só vale com MINIMO_CONCORDANTES sessões iguais no regime (revisão do Codex, 2a volta, achado 3)
    padrao = {}
    for k, v in grupos.items():
        ex, n = Counter(v).most_common(1)[0]
        padrao[k] = ex if n >= MINIMO_CONCORDANTES else None
    cal: dict[date, Sessao] = {}
    fora: dict[date, str] = {}
    d, fim = min(por_dia), max(por_dia)
    while d <= fim:
        if d.weekday() >= 5:
            d += timedelta(days=1)
            continue
        barras = sorted(set(por_dia.get(d, [])))
        if not barras:
            fora[d] = "dia útil sem barra: feriado não comprovado"
        elif padrao[regime(d)] is None:
            fora[d] = f"regime com menos de {MINIMO_CONCORDANTES} sessões concordantes: padrão não comprovado"
        elif extremos[d] != padrao[regime(d)]:
            fora[d] = f"extremidades {extremos[d]} fora do padrão {padrao[regime(d)]} do regime"
        elif any(b2 - b1 != BARRA for b1, b2 in zip(barras, barras[1:])):
            fora[d] = "buraco dentro do pregão"
        else:
            abertura = servidor_para_utc(barras[0])
            fechamento = servidor_para_utc(barras[-1] + BARRA)
            limite = fechamento - MARGEM_ZERAGEM
            if limite <= abertura:
                fora[d] = "pregão curto demais para operar"
            else:
                virada = meia_noite_servidor_utc(d + timedelta(days=1))
                cal[d] = Sessao(simbolo, d, abertura, fechamento, limite, virada, offset_servidor_min(abertura),
                                False, FONTE_INFERIDA, f"{len(barras)} barras, contíguas, no padrão do regime")
        d += timedelta(days=1)
    return cal, fora


def obter_sessao(calendario: Mapping[date, Sessao], simbolo: str, data_servidor: date) -> Sessao:
    try:
        s = calendario[data_servidor]
    except KeyError as exc:
        raise KeyError(f"{simbolo} {data_servidor}: sem calendário comprovável (inapto)") from exc
    if s.simbolo != simbolo:
        raise ValueError("calendário de outro símbolo")
    return s
