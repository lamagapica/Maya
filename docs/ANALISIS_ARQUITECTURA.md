# Mini-Maya: análisis de carencias frente a un DCC profesional

Documento de arquitectura. Basado en la revisión del código (~3.200 líneas) y en mediciones
con `tools/bench_core.py`. Los tiempos absolutos dependen de la máquina; la tendencia no.

## Diagnóstico de partida

La base es sólida para un prototipo: n-gonos reales, Catmull-Clark funcional y tests de variedad
cerrada y volumen. El problema es que la arquitectura está pensada para operaciones sueltas sobre
mallas pequeñas, y un flujo tipo Maya depende de topología persistente, historia y escala.

| Caras (quads) | render_arrays() | face_normals() | edges() |
|---|---|---|---|
| 6.144 | 152 ms | 70 ms | 7 ms |
| 24.576 | 611 ms | 283 ms | 40 ms |
| 98.304 | 2.449 ms | 1.193 ms | 228 ms |

## 1. Estructuras de datos y topología

Actual: `Mesh.f` es lista de listas de Python; `half_edges()`, `edge_faces()` y `edges()` se
reconstruyen en cada llamada (O(F), con ordenación), incluso en cada clic de picking.

Falta:
- Conectividad persistente (twin/next/prev/vert/face). Half-edge en arrays `int32` (solo variedad 2)
  o BMesh / radial-edge (admite aristas con más de 2 caras; recomendado).
- IDs estables de elementos. `compact()` renumera vértices y la selección de aristas son tuplas de
  índices; por eso `op_extrude` borra la selección de vértices y aristas.
- Sistema de atributos: por vértice, arista (crease, sharp, seam), cara (material, smoothing group)
  y esquina (UV reales, normales personalizadas, color). Hoy solo hay `smooth` global y UV procedural
  en `corner_uv()`.
- Edge loops y rings (recorrido por valencia 4; parada en polos, triángulos y bordes), selección de
  loop/ring, shortest path, grow/shrink.
- Triangulación correcta de n-gonos cóncavos (earcut o proyección al plano de Newell), en lugar de abanico.
- Subdivisión: sin creases, sin interpolación de UV/atributos, sin superficie límite. Un vértice de
  borde con ≠2 aristas de borde no se actualiza (`if len(bd) == 2` sin `else`). Integrar OpenSubdiv
  (CPU `Far` + GPU `Osd`).

## 2. Historial de construcción y grafo de dependencias

Actual: `Node` mezcla transformación y malla; las operaciones mutan `n.mesh`; el undo es
`copy.deepcopy` de toda la escena hasta 60 veces; la selección vive en `Viewport`; la lógica de
operaciones está en `main_window.py`.

Falta:
- Separar transform y shape; instancias reales (hoy `duplicate()` hace deepcopy); orden de rotación,
  pivotes, freeze transform.
- DG/DAG con plugs: nodos `extrude -> bevel -> smooth` con `inMesh/outMesh`, flags de dirty,
  evaluación pull.
- Mallas inmutables entre nodos con buffers compartidos (copy-on-write): el undo pasa a ser cambiar punteros.
- Nodo tweak (equivalente a `polyTweak`) para mover vértices sin romper la historia.
- Estrategia ante el problema de nombrado topológico (componentes referenciados aguas arriba).
- Capa de comandos (`Command` con do/undo o journal) independiente de Qt: base para scripting y tests sin GUI.

## 3. Modelado fino e interactivo

Actual:
- Extrude: desplaza por normal promediada normalizada; el grosor real en paredes inclinadas es
  `d·cosθ` (hace falta offset de ingletes `d/cosθ`). Sin inset, divisiones, ni extrusión de
  aristas/vértices/caras individuales.
- Bevel: 1 segmento; el ancho se mide deslizando a lo largo de la arista vecina, no perpendicular
  (solo exacto a 90°). Sin perfil, segmentos, ingletes, clamp, bevel de vértices ni UV.

Falta:
- Multi-cut modal con snapping (vértice, punto medio, 1/N, cuadrícula, ángulo), corte de n-gonos por
  polilínea, "cut through" con X-ray; requiere picking con buffer de profundidad/ID.
- Bevel: offset por ancho/profundidad/porcentaje, N segmentos, perfil superelíptico o curva,
  patrones de inglete en valencia >=3, pesos por arista, harden normals.
- Bridge: loops ordenados con distinto número de aristas, twist, taper, divisiones.
- Extrude paramétrico como nodo del DAG (offset, thickness, divisiones, twist/taper, a lo largo de curva).
- Simetría espacial (KD-tree con tolerancia) y topológica (propagación desde arista semilla).
- Básicos ausentes: merge/weld/collapse, delete vs dissolve, split polygon, insert edge loop, edge
  slide, fill hole, connect, soft selection, marquee/lasso/paint, snapping global, normales duras,
  UV unwrap, booleanos (usar una librería robusta tipo `manifold`).

## 4. Rendimiento

Cuellos de botella medidos:
- `_gpu()` regenera pos+nrm+uv por esquina de triángulo en cada cambio de versión, sin index buffer
  y con bucles Python (`triangles()`, `face_normals()`). Solución: index buffer, escritura parcial
  del buffer, dirty ranges, normales en shader o C++.
- Listas de listas: ~100-200 B por quad frente a 16 B en `int32`. Pasar a arrays planos (offsets + índices).
- Picking por fuerza bruta (`ray_triangles` sobre todos los triángulos, proyección de todos los
  vértices). Usar BVH o picking por ID en GPU.
- C++ (nanobind/pybind11, zero-copy sobre NumPy) para normales, tessellation, adyacencia,
  subdivisión (OpenSubdiv), booleanos, BVH y deformadores. Python queda para UI y orquestación.
- Path tracer: fuerza bruta O(rayos x triángulos) con AABB solo por objeto; un BVH (SAH) cambia el
  orden de magnitud. Prioridad menor que el modelado.

## 5. Deformadores y rigging

Actual: inexistente; la malla guarda solo `v`, sin separar base y deformada.

Falta:
- Pipeline de evaluación con orden de deformadores (`inMesh -> deformer -> outMesh` en el DAG).
- Pesos por vértice (pintura, suavizado, normalización, mirror).
- Lattice (FFD), Bend/Twist/Flare, Cluster, soft selection.
- Joints con orient, bind pose, LBS y Dual Quaternion; luego blend shapes y correctivos.
- Skinning en vertex/compute shader (matrices en SSBO) por encima de decenas de miles de vértices.

## Orden recomendado

1. Mallas SoA con half-edge/BMesh persistente, IDs estables y atributos.
2. Capa de comandos + mallas inmutables + undo por punteros.
3. Buffers GPU parciales y picking por ID.
4. Kernel C++ con nanobind y OpenSubdiv.
5. DAG completo con nodos de operación y tweak; Extrude/Bevel como nodos paramétricos.
6. Loops, multi-cut, bridge, simetría.
7. Deformadores y skinning.
