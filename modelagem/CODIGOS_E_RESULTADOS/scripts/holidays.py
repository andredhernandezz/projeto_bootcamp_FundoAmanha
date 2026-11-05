"""Reproducible holidays for the observed 2025-26 window: federal + two municipal Juiz de Fora days.
The holiday lists are not the steel plant's actual staffing/shift calendar.
Sources: official MG calendar and TRT3 local holiday lists.
"""
from datetime import date
F = {2025:{(1,1):'Confraternizacao Universal',(4,18):'Sexta-feira Santa',(4,21):'Tiradentes',(5,1):'Dia do Trabalho',(9,7):'Independencia',(10,12):'N.S. Aparecida',(11,2):'Finados',(11,15):'Proclamacao',(11,20):'Consciencia Negra',(12,25):'Natal',(6,19):'Corpus Christi - Juiz de Fora',(6,13):'Santo Antonio - Juiz de Fora'},2026:{(1,1):'Confraternizacao Universal',(4,3):'Sexta-feira Santa',(4,21):'Tiradentes',(5,1):'Dia do Trabalho',(9,7):'Independencia',(10,12):'N.S. Aparecida',(11,2):'Finados',(11,15):'Proclamacao',(11,20):'Consciencia Negra',(12,25):'Natal',(6,4):'Corpus Christi - Juiz de Fora',(6,13):'Santo Antonio - Juiz de Fora'}}
def Brazil(subdiv=None,years=None):
    return {date(y,m,d):name for y in (years or [2025,2026]) for (m,d),name in F[y].items()}
