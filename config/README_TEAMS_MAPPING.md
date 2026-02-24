# Mapeo de nombres de equipos

## ¿Para qué sirve?

Los mismos clubes aparecen con nombres distintos en las fuentes (ej: "Sporting CP" vs "Sporting Clube de Portugal", "Bayern München" vs "FC Bayern München"). El archivo `teams_mapping.csv` unifica estos alias en un nombre canónico para que:

- **matches** y **teams** usen el mismo nombre
- **all_teams.csv** no tenga duplicados del mismo club
- Las listas de equipos por temporada/liga sean coherentes

## Cómo añadir correlaciones

1. Edita `config/teams_mapping.csv`
2. Añade una fila: `alias,canonical`
   - `alias`: nombre que aparece en los datos
   - `canonical`: nombre oficial/unificado que quieres usar
3. Ejemplo: `Sporting CP,Sporting Clube de Portugal`
4. Regenera los datos:
   ```bash
   python3 clean_champions_data.py
   python3 unify_teams.py
   ```

## Cómo descubrir duplicados

```bash
# Lista todos los nombres únicos
python3 discover_team_aliases.py

# Sugiere pares similares (revisar manualmente: algunos son falsos positivos)
python3 discover_team_aliases.py --similar
```

**Cuidado:** No todos los pares similares son el mismo equipo (ej: Randers FC ≠ Rangers FC). Revisa antes de añadir al mapping.
