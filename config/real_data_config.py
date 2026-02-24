#!/usr/bin/env python3
"""
CONFIGURACIÓN PARA DATOS REALES CHAMPIONS LEAGUE
"""

# Opción 1: API-Football (requiere key real)
API_FOOTBALL_KEY = "TU_API_FOOTBALL_KEY_AQUI"  # Reemplazar con tu key real

# Opción 2: TheSportsDB (alternativa gratuita)
# TheSportsDB ofrece datos de fútbol gratuitos con características básicas
USE_SPORTSDB = True

# Opción 3: Statorium (alternativa gratuita)
# Statorium ofrece widgets de fútbol gratuitos
USE_STADIUM = False

# Opción 4: iSports (alternativa gratuita)
# iSports ofrece datos de múltiples deportes incluyendo fútbol
USE_ISPORTS = False

# Configuración según la opción seleccionada
if USE_SPORTSDB:
    print("🌍 Usando TheSportsDB (alternativa gratuita)")
    # TheSportsDB no requiere API key
    API_FOOTBALL_KEY = None
    API_FOOTBALL_BASE_URL = None
elif USE_STADIUM:
    print("🌍 Usando Stadium (alternativa gratuita)")
    # Stadium no requiere API key para widgets básicos
    API_FOOTBALL_KEY = None
    API_FOOTBALL_BASE_URL = None
elif USE_ISPORTS:
    print("🌍 Usando iSports (alternativa gratuita)")
    # iSports requiere key pero tiene más datos
    API_FOOTBALL_KEY = "TU_API_ISPORTS_KEY"
    API_FOOTBALL_BASE_URL = "https://v3.football.api-sports.io"
else:
    print("🌍 Usando API-Football (requiere key real)")
    # API-Football requiere key real para datos completos
    API_FOOTBALL_KEY = "TU_API_FOOTBALL_KEY_AQUI"  # Reemplazar con tu key real
    API_FOOTBALL_BASE_URL = "https://v3.football.api-sports.io"

print(f"✅ Configuración lista para datos reales")
print(f"   API-Football Key: {'Configurada' if API_FOOTBALL_KEY else 'No configurada'}")
print(f"   TheSportsDB: {'Habilitado' if USE_SPORTSDB else 'No habilitado'}")
print(f"   Stadium: {'Habilitado' if USE_STADIUM else 'No habilitado'}")
print(f"   iSports: {'Habilitado' if USE_ISPORTS else 'No habilitado'}")
print()
print("🔗 INSTRUCCIONES PARA DATOS REALES:")
print("1. API-Football: Datos completos oficiales (requiere key)")
print("2. TheSportsDB: Datos básicos gratuitos (sin key requerida)")
print("3. Stadium: Widgets de fútbol gratuitos (sin key requerida)")
print("4. iSports: Múltiples deportes (requiere key)")
print()
print("🎯 Para cambiar la configuración, edita este archivo y reemplaza la variable correspondiente.")