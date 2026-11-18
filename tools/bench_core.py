import time, sys, tracemalloc
sys.path.insert(0,'.')
import numpy as np
from minimaya.core import primitives as P
from minimaya.core.modeling import catmull_clark, extrude_faces, bevel_edges

def T(label, fn):
    t=time.perf_counter(); r=fn(); dt=time.perf_counter()-t
    print(f"  {label:34s} {dt*1000:9.1f} ms"); return r

for lv in (2,3,4):
    m=P.cube(2.0)
    tracemalloc.start()
    t=time.perf_counter(); catmull_clark(m,lv); dt=time.perf_counter()-t
    cur,peak=tracemalloc.get_traced_memory(); tracemalloc.stop()
    print(f"\nCube + CatmullClark nivel {lv}: {len(m.f):,} caras, {len(m.v):,} verts  | subdividir: {dt*1000:.0f} ms | pico mem {peak/1e6:.0f} MB")
    T("render_arrays() (rebuild GPU)", m.render_arrays)
    T("edges() (sorted, por click)", m.edges)
    T("face_normals()", m.face_normals)
    T("vertex_normals()", m.vertex_normals)
    T("half_edges()", m.half_edges)
    T("copy()  (undo snapshot basico)", m.copy)
    import copy as C
    T("deepcopy(mesh) (como Scene.state)", lambda: C.deepcopy(m))
    top=[i for i,n in enumerate(m.face_normals()) if n[1]>0.9][:200]
    m2=m.copy(); T(f"extrude {len(top)} caras", lambda: extrude_faces(m2,top,0.1))
    m3=m.copy(); es=m3.edges()[:2000]; T(f"bevel {len(es)} aristas", lambda: bevel_edges(m3,es,0.01))
