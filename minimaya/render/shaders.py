"""Shaders GLSL 4.10 core (compatibles con macOS, que limita OpenGL a 4.1)."""

PBR_VS = """
#version 410 core
in vec3 in_pos; in vec3 in_nrm; in vec2 in_uv;
uniform mat4 uModel; uniform mat4 uVP; uniform mat3 uNMat; uniform mat4 uLightVP;
out vec3 vWorld; out vec3 vN; out vec2 vUV; out vec4 vLP;
void main() {
    vec4 w = uModel * vec4(in_pos, 1.0);
    vWorld = w.xyz;
    vN = normalize(uNMat * in_nrm);
    vUV = in_uv;
    vLP = uLightVP * w;
    gl_Position = uVP * w;
}
"""

PBR_FS = """
#version 410 core
in vec3 vWorld; in vec3 vN; in vec2 vUV; in vec4 vLP;
uniform vec3 uCam;
uniform vec3 uAlbedo; uniform float uRough; uniform float uMetal;
uniform int uHasAlbedo; uniform int uHasNormal; uniform float uNormalStrength;
uniform int uShadowOn; uniform int uShadowIdx;
uniform int uNLights; uniform vec4 uLPos[8]; uniform vec4 uLCol[8];
uniform float uEnvInt; uniform float uEnvMaxLod;
uniform sampler2D uAlbedoTex; uniform sampler2D uNormalTex;
uniform sampler2D uShadow; uniform sampler2D uEnv; uniform sampler2D uIrr;
out vec4 fragColor;
const float PI = 3.14159265359;

vec2 dirToUV(vec3 d) { return vec2(atan(d.z, d.x) / (2.0 * PI) + 0.5, acos(clamp(d.y, -1.0, 1.0)) / PI); }

float shadowFactor(float ndl) {
    vec3 pc = vLP.xyz / vLP.w * 0.5 + 0.5;
    if (pc.z > 1.0 || pc.x < 0.0 || pc.x > 1.0 || pc.y < 0.0 || pc.y > 1.0) return 1.0;
    float bias = max(0.0035 * (1.0 - ndl), 0.0007);
    vec2 ts = 1.0 / vec2(textureSize(uShadow, 0));
    float s = 0.0;
    for (int x = -1; x <= 1; x++)
        for (int y = -1; y <= 1; y++)
            s += (pc.z - bias <= texture(uShadow, pc.xy + vec2(x, y) * ts).r) ? 1.0 : 0.0;
    return s / 9.0;
}

vec3 perturbNormal(vec3 N, vec3 p, vec2 uv) {      // marco de cotangentes sin tangentes por vértice
    vec3 dp1 = dFdx(p), dp2 = dFdy(p);
    vec2 duv1 = dFdx(uv), duv2 = dFdy(uv);
    vec3 dp2perp = cross(dp2, N), dp1perp = cross(N, dp1);
    vec3 T = dp2perp * duv1.x + dp1perp * duv2.x;
    vec3 B = dp2perp * duv1.y + dp1perp * duv2.y;
    float invmax = inversesqrt(max(max(dot(T, T), dot(B, B)), 1e-12));
    vec3 m = texture(uNormalTex, uv).xyz * 2.0 - 1.0;
    m.xy *= uNormalStrength;
    return normalize(mat3(T * invmax, B * invmax, N) * m);
}

float D_GGX(float NdH, float a) { float a2 = a * a; float d = NdH * NdH * (a2 - 1.0) + 1.0; return a2 / (PI * d * d); }
float G_Schlick(float NdX, float k) { return NdX / (NdX * (1.0 - k) + k); }
vec3 F_Schlick(vec3 F0, float c) { return F0 + (1.0 - F0) * pow(1.0 - c, 5.0); }
vec3 envBRDF(vec3 F0, float r, float NdV) {          // aproximación de Karis
    const vec4 c0 = vec4(-1.0, -0.0275, -0.572, 0.022);
    const vec4 c1 = vec4(1.0, 0.0425, 1.04, -0.04);
    vec4 rr = r * c0 + c1;
    float a004 = min(rr.x * rr.x, exp2(-9.28 * NdV)) * rr.x + rr.y;
    vec2 AB = vec2(-1.04, 1.04) * a004 + rr.zw;
    return F0 * AB.x + AB.y;
}

void main() {
    vec3 N = normalize(vN);
    if (!gl_FrontFacing) N = -N;
    vec3 V = normalize(uCam - vWorld);
    vec3 albedo = pow(uAlbedo, vec3(2.2));
    if (uHasAlbedo == 1) albedo *= pow(texture(uAlbedoTex, vUV).rgb, vec3(2.2));
    if (uHasNormal == 1) N = perturbNormal(N, vWorld, vUV);
    float rough = clamp(uRough, 0.04, 1.0);
    float metal = uMetal;
    vec3 F0 = mix(vec3(0.04), albedo, metal);
    float NdV = max(dot(N, V), 1e-4);
    float a = rough * rough, k = (rough + 1.0) * (rough + 1.0) / 8.0;

    vec3 Lo = vec3(0.0);
    for (int i = 0; i < uNLights; i++) {
        vec3 L; float att = 1.0;
        if (uLPos[i].w < 0.5) { L = normalize(uLPos[i].xyz); }
        else { vec3 d = uLPos[i].xyz - vWorld; float dist = length(d); L = d / dist; att = 1.0 / (dist * dist + 1e-4); }
        float NdL = max(dot(N, L), 0.0);
        if (NdL <= 0.0) continue;
        vec3 H = normalize(V + L);
        float NdH = max(dot(N, H), 0.0), VdH = max(dot(V, H), 0.0);
        vec3 F = F_Schlick(F0, VdH);
        vec3 spec = D_GGX(NdH, a) * G_Schlick(NdV, k) * G_Schlick(NdL, k) * F / (4.0 * NdV * NdL + 1e-4);
        vec3 kd = (1.0 - F) * (1.0 - metal);
        float sh = 1.0;
        if (uShadowOn == 1 && i == uShadowIdx) sh = shadowFactor(NdL);
        Lo += (kd * albedo / PI + spec) * uLCol[i].rgb * att * NdL * sh;
    }

    vec3 Fa = F_Schlick(F0, NdV);
    vec3 irr = texture(uIrr, dirToUV(N)).rgb;
    vec3 R = reflect(-V, N);
    vec3 pref = textureLod(uEnv, dirToUV(R), rough * uEnvMaxLod).rgb;
    vec3 ambient = ((1.0 - Fa) * (1.0 - metal) * albedo * irr + pref * envBRDF(F0, rough, NdV)) * uEnvInt;

    vec3 c = Lo + ambient;
    c = (c * (2.51 * c + 0.03)) / (c * (2.43 * c + 0.59) + 0.14);   // ACES
    fragColor = vec4(pow(clamp(c, 0.0, 1.0), vec3(1.0 / 2.2)), 1.0);
}
"""

SHADOW_VS = """
#version 410 core
in vec3 in_pos; uniform mat4 uMVP;
void main() { gl_Position = uMVP * vec4(in_pos, 1.0); }
"""
SHADOW_FS = """
#version 410 core
void main() {}
"""

LINE_VS = """
#version 410 core
in vec3 in_pos; uniform mat4 uMVP; uniform float uBias; uniform float uPointSize;
void main() { gl_Position = uMVP * vec4(in_pos, 1.0); gl_Position.z -= uBias * gl_Position.w; gl_PointSize = uPointSize; }
"""
LINE_FS = """
#version 410 core
uniform vec4 uColor; uniform int uRound; out vec4 f;
void main() {
    if (uRound == 1 && length(gl_PointCoord - vec2(0.5)) > 0.5) discard;
    f = uColor;
}
"""

VCOL_VS = """
#version 410 core
in vec3 in_pos; in vec3 in_col; uniform mat4 uMVP; out vec3 vc;
void main() { gl_Position = uMVP * vec4(in_pos, 1.0); vc = in_col; }
"""
VCOL_FS = """
#version 410 core
in vec3 vc; out vec4 f;
void main() { f = vec4(vc, 1.0); }
"""

SOLID_VS = """
#version 410 core
in vec3 in_pos; in vec3 in_nrm; uniform mat4 uMVP; uniform mat3 uNV; out vec3 n;
void main() { gl_Position = uMVP * vec4(in_pos, 1.0); n = normalize(uNV * in_nrm); }
"""
SOLID_FS = """
#version 410 core
in vec3 n; uniform vec4 uColor; out vec4 f;
void main() { float l = 0.55 + 0.45 * abs(n.z); f = vec4(uColor.rgb * l, uColor.a); }
"""
