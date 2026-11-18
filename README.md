# Mini-Maya

Editor 3D de escritorio (Python + PySide6 + ModernGL) con modelado poligonal, materiales PBR/UV
y render en tiempo real + path tracer offline.

## Ejecución
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt                        # Python 3.11 o 3.12
python main.py
python -m tests.test_core                              # pruebas del núcleo (sin GUI)
```
Requiere GPU con OpenGL 4.1+ (macOS está limitado a 4.1; en Windows/Linux el driver suele dar 4.5+).
En Linux, si el gestor de ventanas captura Alt+arrastrar, desactiva ese atajo o usa Meta.

## Controles
| Acción | Atajo |
|---|---|
| Orbitar / Panear / Zoom | Alt+LMB / Alt+MMB / Alt+RMB o rueda |
| Seleccionar / Mover / Rotar / Escalar | Q / W / E / R |
| Objeto / Vértice / Arista / Cara | F8 / F9 / F10 / F11 (Shift = añadir/quitar) |
| Wireframe / Shaded / Textured / Wire on Shaded | 4 / 5 / 6 / 7 |
| Encuadrar, Duplicar, Agrupar, Borrar | F, Ctrl+D, Ctrl+G, Supr |
| Extrude, Bevel, Smooth (Catmull-Clark) | Ctrl+E, Ctrl+B, 3 |
| Render Frame | Ctrl+R |

Flujo de Extrude: F11 → clic en una cara → Ctrl+E → W para arrastrar las caras nuevas.
Los parámetros (distancia, ancho de bisel, niveles) están en Attribute Editor > Channel Box.

## Estructura
```
main.py
minimaya/core/    mathutils, camera, mesh (caras + semiaristas derivadas), modeling (extrude, bevel,
                  Catmull-Clark), primitives, scene (nodos, jerarquía, materiales, undo), meshio (OBJ/glTF)
minimaya/render/  shaders (GLSL PBR/sombras), viewport (OpenGL), gizmo, env (HDRI), pathtracer (CPU)
minimaya/ui/      main_window, outliner, channel_box, material_editor, uv_editor, render_view, theme, icons
tests/test_core.py
```

## Límites conocidos
- FBX no se soporta (formato propietario): convierte a glTF/OBJ. glTF/GLB se importa con `trimesh`.
- Bevel: 1 segmento (chaflán); pensado para mallas cerradas.
- Los polígonos se triangulan en abanico (válido para n-gonos convexos).
- El UV Editor aplica proyección + escala/offset globales; no hay edición de islas por vértice.
- Path tracer en NumPy sin BVH: usa resoluciones bajas y escenas de pocos miles de triángulos.
- IBL en el viewport usa mipmaps del HDRI y una irradiancia de baja resolución (aproximación).

## Documentación
- `docs/ANALISIS_ARQUITECTURA.md`: análisis de carencias y hoja de ruta.
- `tools/bench_core.py`: benchmark del núcleo de mallas (`python tools/bench_core.py`).
