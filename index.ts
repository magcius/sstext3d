
import * as opentype from 'opentype.js';
import * as poly2tri from 'poly2tri';
import { mat4 } from 'gl-matrix';

//@ts-ignore
import { readFileSync } from 'fs';

//@ts-ignore
const IS_DEVELOPMENT: boolean = process.env.NODE_ENV === 'development';

const DEBUG = IS_DEVELOPMENT;

function leftPad(S: string, spaces: number, ch: string = '0'): string {
    while (S.length < spaces)
        S = `${ch}${S}`;
    return S;
}

function createTextProgram(gl: WebGLRenderingContext): WebGLProgram {
    const vert = `
precision mediump float;

attribute vec3 a_Position;
attribute vec3 a_Normal;

uniform mat4 u_ClipFromView;
uniform mat4 u_ViewFromLocal;
uniform mat4 u_ViewFromLocalNormal;

struct Material {
    vec3 AmbientReflectance;
    vec3 DiffuseReflectance;
    vec4 SpecularReflectance; // shininess in W
};

struct Light {
    vec3 Position;
    vec3 AmbientIntensity;
    vec3 DiffuseIntensity;
    vec3 SpecularIntensity;
};

struct LightResult {
    vec3 AmbientColor;
    vec3 DiffuseColor;
    vec3 SpecularColor;
};

varying vec3 v_AmbientColor;
varying vec3 v_DiffuseColor;
varying vec3 v_SpecularColor;
varying vec2 v_TexCoord;

// The classic OpenGL lighting model.
void EvalLight(inout LightResult t_LightResult, in Material t_Material, in Light t_Light, in vec3 t_PositionView, in vec3 t_Normal)
{
    // Definition of view space.
    vec3 t_VertexToEye = vec3(0.0, 0.0, 0.0) - t_PositionView;
    vec3 t_VertexToLight = normalize(t_Light.Position - t_PositionView);
    vec3 t_H = normalize(t_VertexToEye + t_VertexToLight);

    t_LightResult.AmbientColor += t_Material.AmbientReflectance * t_Light.AmbientIntensity;
    t_LightResult.DiffuseColor += t_Material.DiffuseReflectance * t_Light.DiffuseIntensity * max(dot(t_Normal, t_VertexToLight), 0.0);
    t_LightResult.SpecularColor += t_Material.SpecularReflectance.xyz * t_Light.SpecularIntensity * pow(max(dot(t_Normal, t_H), 0.0), t_Material.SpecularReflectance.w);
}

void main() {
    vec4 t_PositionView = (u_ViewFromLocal * vec4(a_Position, 1.0));
    vec3 t_Normal = normalize((u_ViewFromLocalNormal * vec4(a_Normal, 0.0)).xyz);

    gl_Position = u_ClipFromView * t_PositionView;

    Material t_Material;
    t_Material.AmbientReflectance  = vec3(0.2, 0.2, 0.2);
    t_Material.DiffuseReflectance  = vec3(1.0, 1.0, 1.0);
    t_Material.SpecularReflectance = vec4(0.5, 0.5, 0.5, 32.0);

    Light t_Light;
    LightResult t_LightResult;

    // Full-scene ambient.
    t_LightResult.AmbientColor += t_Material.AmbientReflectance * vec3(1.0, 1.0, 1.0);

    // Light 1.
    t_Light.AmbientIntensity = vec3(0.2);
    t_Light.DiffuseIntensity = vec3(0.7);
    t_Light.SpecularIntensity = vec3(1.0);
    t_Light.Position = vec3(0.0, 25.0, 150.0);
    EvalLight(t_LightResult, t_Material, t_Light, t_PositionView.xyz, t_Normal);

    // Light 2.
    t_Light.AmbientIntensity = vec3(0.1);
    t_Light.DiffuseIntensity = vec3(0.7);
    t_Light.SpecularIntensity = vec3(1.0);
    t_Light.Position = vec3(25.0, 150.0, 50.0);
    EvalLight(t_LightResult, t_Material, t_Light, t_PositionView.xyz, t_Normal);

    v_AmbientColor = t_LightResult.AmbientColor;
    v_DiffuseColor = t_LightResult.DiffuseColor;
    v_SpecularColor = t_LightResult.SpecularColor;
    v_TexCoord = t_Normal.xy * 0.5 + 0.5;
}
`;

    const frag = `
precision mediump float;

uniform sampler2D s_ReflectionMap;

varying vec3 v_AmbientColor;
varying vec3 v_DiffuseColor;
varying vec3 v_SpecularColor;
varying vec2 v_TexCoord;

void main() {
    vec4 t_ReflSample = texture2D(s_ReflectionMap, v_TexCoord);
    vec3 t_ColorMul = v_DiffuseColor + v_AmbientColor;
    vec3 t_ColorAdd = v_SpecularColor;
    gl_FragColor = vec4(t_ColorMul.rgb * t_ReflSample.rgb + t_ColorAdd.rgb, 1.0);
}
`;

    return compileProgram(gl, vert, frag);
}

function assert(b: boolean, message: string = ""): void {
    if (!b) { console.error(new Error().stack); throw new Error(`Assert fail: ${message}`); }
}

function assertExists<T>(v: T | null | undefined): T {
    if (v !== undefined && v !== null)
        return v;
    else
        throw new Error("Missing object");
}

function prependLineNo(str: string, lineStart: number = 1) {
    const lines = str.split('\n');
    return lines.map((s, i) => `${leftPad('' + (lineStart + i), 4, ' ')}  ${s}`).join('\n');
}

function compileShader(gl: WebGLRenderingContext, str: string, type: number): WebGLShader | null {
    const shader: WebGLShader = assertExists(gl.createShader(type));

    gl.shaderSource(shader, str);
    gl.compileShader(shader);

    if (DEBUG && !gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        console.error(prependLineNo(str));
        const debug_shaders = gl.getExtension('WEBGL_debug_shaders');
        if (debug_shaders)
            console.error(debug_shaders.getTranslatedShaderSource(shader));
        console.error(gl.getShaderInfoLog(shader));
        gl.deleteShader(shader);
        return null;
    }

    return shader;
}

function compileProgram(gl: WebGLRenderingContext, vert: string, frag: string): WebGLProgram | null {
    const vertShader = compileShader(gl, vert, gl.VERTEX_SHADER);
    const fragShader = compileShader(gl, frag, gl.FRAGMENT_SHADER);
    if (!vertShader || !fragShader)
        return null;
    const prog = gl.createProgram();
    gl.attachShader(prog, vertShader);
    gl.attachShader(prog, fragShader);
    gl.linkProgram(prog);
    gl.deleteShader(vertShader);
    gl.deleteShader(fragShader);
    if (DEBUG && !gl.getProgramParameter(prog, gl.LINK_STATUS)) {
        console.error(vert);
        console.error(frag);
        console.error(gl.getProgramInfoLog(prog));
        gl.deleteProgram(prog);
        return null;
    }
    return prog;
}

//#region ot
const FONT_SIZE = 512;

function getPointBezierCubic(p0: number, p1: number, p2: number, p3: number, t: number): number {
    const cf0 = (p0 * -1) + (p1 *  3) + (p2 * -3) +  (p3 *  1);
    const cf1 = (p0 *  3) + (p1 * -6) + (p2 *  3) +  (p3 *  0);
    const cf2 = (p0 * -3) + (p1 *  3) + (p2 *  0) +  (p3 *  0);
    const cf3 = (p0 *  1) + (p1 *  0) + (p2 *  0) +  (p3 *  0);
    return (((cf0 * t + cf1) * t + cf2) * t + cf3);
}

function getPointBezierQuadratic(p0: number, p1: number, p2: number, t: number): number {
    return (p0 * (1 - t)*(1 - t)) + (p1 * 2*t*(1 - t)) + p2*t*t;
}

type bezcb = (x: number, y: number) => void;

const giant = 16;

// todo: castle
function evalBezierCubic2d(p0: Point, p1: Point, p2: Point, p3: Point, callback: bezcb, steps = giant): void {
    const ms = steps - 1;
    for (let i = 0; i < steps; i++) {
        const t = i/ms;
        const x = getPointBezierCubic(p0.x, p1.x, p2.x, p3.x, t);
        const y = getPointBezierCubic(p0.y, p1.y, p2.y, p3.y, t);
        callback(x, y);
    }
}

function evalBezierQuadratic2d(p0: Point, p1: Point, p2: Point, callback: bezcb, steps = giant): void {
    const ms = steps - 1;
    for (let i = 0; i < steps; i++) {
        const t = i/ms;
        const x = getPointBezierQuadratic(p0.x, p1.x, p2.x, t);
        const y = getPointBezierQuadratic(p0.y, p1.y, p2.y, t);
        callback(x, y);
    }
}

function mod(a: number, b: number): number {
    return (a + b) % b;
}

function modi<T>(L: T[], i: number): T {
    return L[mod(i, L.length)];
}

type Point = { x: number, y: number };

class BBOX {
    public minX: number = Infinity;
    public minY: number = Infinity;
    public maxX: number = -Infinity;
    public maxY: number = -Infinity;
}

function bboxReset(dst: BBOX): void {
    dst.minX = Infinity;
    dst.minY = Infinity;
    dst.maxX = -Infinity;
    dst.maxY = -Infinity;
}

function bboxUnionPoint(dst: BBOX, x: number, y: number): void {
    dst.minX = Math.min(dst.minX, x);
    dst.minY = Math.min(dst.minY, y);
    dst.maxX = Math.max(dst.maxX, x);
    dst.maxY = Math.max(dst.maxY, y);
}

function bboxTestPoint(b: BBOX, x: number, y: number): boolean {
    return (
        x >= b.minX && x <= b.maxX &&
        y >= b.minY && y <= b.maxY
    );
}

class GlyphDataBucket {
    public bbox = new BBOX();
    // Each bucket is allowed one main shape, with multiple holes.
    public loops: Point[][] = [];
}

class GlyphData {
    public vertexBuffer: WebGLBuffer;
    public indexBuffer: WebGLBuffer;
    public indexCount: number;

    constructor(gl: WebGLRenderingContext, private glyph: opentypejs.Glyph, private path: opentypejs.Path) {
        // Convert to path data.

        // TODO(jstpierre): This is a giant hack.

        // A stash of all points.
        const points: Point[] = [];

        const buckets: GlyphDataBucket[] = [];
        let currentBucket: GlyphDataBucket | null = null;
        let currentPoints: Point[] | null = null;

        function point(x: number, y: number) {
            if (currentBucket === null) {
                currentPoints = null;

                // First, test to see if this point is in any existing buckets.
                for (let i = 0; i < buckets.length; i++) {
                    if (bboxTestPoint(buckets[i].bbox, x, y)) {
                        currentBucket = buckets[i];
                        break;
                    }
                }

                // If no bucket fits, create a new bucket...
                if (currentBucket === null) {
                    currentBucket = new GlyphDataBucket();
                    buckets.push(currentBucket);
                }
            }

            if (currentPoints === null) {
                currentPoints = [];
                currentBucket.loops.push(currentPoints);
            }

            // First, check to make sure we're not adding a repeat point. If so, drop it.
            // This is a dumb hack.
            if (currentPoints.length > 0) {
                const h = currentPoints[0], t = currentPoints[currentPoints.length - 1];
                if (h.x === x && h.y === y)
                    return;
                if (t.x === x && t.y === y)
                    return;
            }

            bboxUnionPoint(currentBucket.bbox, x, y);
            const p: Point = { x, y };
            currentPoints.push(p);
            points.push(p);
        }

        for (let i = 0; i < path.commands.length; i++) {
            const cmd = path.commands[i];
            if (cmd.type === 'M') {
                // Begin a new path. Push a new point.
                currentBucket = null;
                point(cmd.x, cmd.y);
            } else if (cmd.type === 'L') {
                // Linear segment.
                point(cmd.x, cmd.y);
            } else if (cmd.type === 'C') {
                // Cubic bezier segment. Evaluate.
                const p0: poly2tri.IPointLike = currentPoints[currentPoints.length - 1];
                const p1: poly2tri.IPointLike = { x: cmd.x1, y: cmd.y1 };
                const p2: poly2tri.IPointLike = { x: cmd.x2, y: cmd.y2 };
                const p3: poly2tri.IPointLike = { x: cmd.x, y: cmd.y };
                evalBezierCubic2d(p0, p1, p2, p3, point);
            } else if (cmd.type === 'Q') {
                // Quadratic bezier segment. Evaluate.
                const p0 = currentPoints[currentPoints.length - 1];
                const p1: poly2tri.IPointLike = { x: cmd.x1, y: cmd.y1 };
                const p2: poly2tri.IPointLike = { x: cmd.x, y: cmd.y };
                evalBezierQuadratic2d(p0, p1, p2, point);
            } else if (cmd.type === 'Z') {
                // Close the current path.
                currentBucket = null;
            } else {
                throw "whoops";
            }
        }

        // front, extr 1, extr 2, back
        const numPoints = points.length * 4;
        const floatsPerPoint = 6; // pos, nrm
        const dataPointChunkOffs = points.length * floatsPerPoint;
        const data = new Float32Array(numPoints * floatsPerPoint);

        for (let i = 0; i < buckets.length; i++) {
            const loops = buckets[i].loops;
            for (let j = 0; j < loops.length; j++) {
                const loop = loops[j];
                for (let k = 0; k < loop.length; k++) {
                    const p = loop[k];
                    const pidx = points.indexOf(p);

                    // Build front-facing point.
                    let idx0 = (pidx * floatsPerPoint) + dataPointChunkOffs * 0;
                    data[idx0++] = p.x;
                    data[idx0++] = p.y;
                    data[idx0++] = 0;
                    data[idx0++] = 0;
                    data[idx0++] = 0;
                    data[idx0++] = 1;

                    // Build back-facing point.
                    let idx1 = (pidx * floatsPerPoint) + dataPointChunkOffs * 1;
                    data[idx1++] = p.x;
                    data[idx1++] = p.y;
                    data[idx1++] = -1;
                    data[idx1++] = 0;
                    data[idx1++] = 0;
                    data[idx1++] = -1;

                    // Calculate the previous and next points to calculate our slope.
                    const pprev = modi(loop, k - 1);
                    const pnext = modi(loop, k + 1);

                    const dx = ((pnext.x - pprev.x));
                    const dy = ((pnext.y - pprev.y));
                    let nx = dy, ny = dx;
                    // normalize
                    const dist = Math.hypot(nx, ny);
                    nx /= dist;
                    ny /= dist;

                    // Build extrusion point 1 (front).
                    let idx2 = (pidx * floatsPerPoint) + dataPointChunkOffs * 2;
                    data[idx2++] = p.x;
                    data[idx2++] = p.y;
                    data[idx2++] = 0;
                    data[idx2++] = nx;
                    data[idx2++] = ny;
                    data[idx2++] = 0;

                    // Build extrusion point 1 (back).
                    let idx3 = (pidx * floatsPerPoint) + dataPointChunkOffs * 3;
                    data[idx3++] = p.x;
                    data[idx3++] = p.y;
                    data[idx3++] = -1;
                    data[idx3++] = nx;
                    data[idx3++] = ny;
                    data[idx3++] = 0;
                }
            }
        }

        this.vertexBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, this.vertexBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);

        // Triangulate and create index data.

        const indexOffsFront = points.length * 0;
        const indexOffsBack  = points.length * 1;
        const indexOffsExtr1 = points.length * 2;
        const indexOffsExtr2 = points.length * 3;

        const indexData: number[] = [];
        for (let i = 0; i < buckets.length; i++) {
            const bucket = buckets[i];
            const sweepContext = new poly2tri.SweepContext(bucket.loops[0]);
            sweepContext.addHoles(bucket.loops.slice(1));
            sweepContext.triangulate();
            const triangles = sweepContext.getTriangles();

            for (let j = 0; j < triangles.length; j++) {
                const i0 = points.indexOf(triangles[j].getPoint(0));
                const i1 = points.indexOf(triangles[j].getPoint(1));
                const i2 = points.indexOf(triangles[j].getPoint(2));
                // Front face
                indexData.push(indexOffsFront + i0);
                indexData.push(indexOffsFront + i1);
                indexData.push(indexOffsFront + i2);
                // Back face.
                indexData.push(indexOffsBack + i0);
                indexData.push(indexOffsBack + i2);
                indexData.push(indexOffsBack + i1);
            }

            // Now do the extrusion faces. We simply iterate over each point and connect the fronts and backs.
            for (let j = 0; j < bucket.loops.length; j++) {
                const loop = bucket.loops[j];
                for (let k = 0; k < loop.length; k++) {
                    const i0 = points.indexOf(modi(loop, k - 1));
                    const i1 = points.indexOf(modi(loop, k - 0));
                    // Extrusion triangle one.
                    indexData.push(indexOffsExtr1 + i0);
                    indexData.push(indexOffsExtr1 + i1);
                    indexData.push(indexOffsExtr2 + i1);
                    // Extrusion triangle two.
                    indexData.push(indexOffsExtr2 + i1);
                    indexData.push(indexOffsExtr2 + i0);
                    indexData.push(indexOffsExtr1 + i0);
                }
            }
        }
        this.indexCount = indexData.length;

        this.indexBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.indexBuffer);
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indexData), gl.STATIC_DRAW);
    }

    public destroy(gl: WebGLRenderingContext): void {
        gl.deleteBuffer(this.vertexBuffer);
        gl.deleteBuffer(this.indexBuffer);
    }
}

class FontData {
    private glyphData: (GlyphData | null)[] = [];

    constructor(public font: opentypejs.Font) {
    }

    public ensureGlyph(gl: WebGLRenderingContext, idx: number): void {
        if (this.glyphData[idx] === undefined) {
            const glyph = this.font.glyphs.get(idx);
            const path = glyph.getPath(0, 0, FONT_SIZE);
            if (path.commands.length === 0)
                this.glyphData[idx] = null;
            else
                this.glyphData[idx] = new GlyphData(gl, glyph, path);
        }
    }

    public getGlyph(idx: number): GlyphData | null {
        assert(this.glyphData[idx] !== undefined);
        return this.glyphData[idx];
    }

    public destroy(gl: WebGLRenderingContext): void {
        for (let i = 0; i < this.glyphData.length; i++)
            if (this.glyphData[i] !== undefined)
                this.glyphData[i].destroy(gl);
    }
}

interface PlacedGlyph {
    x: number;
    y: number;
    glyphIndex: number;
}

function computeNormalMatrix(dst: mat4, m: mat4): void {
    if (dst !== m)
        mat4.copy(dst, m);
    dst[12] = 0;
    dst[13] = 0;
    dst[14] = 0;
    mat4.invert(dst, dst);
    mat4.transpose(dst, dst);
}
//#endregion

const scratchMatrix = mat4.create();
class TextRenderer {
    private placedGlyphs: PlacedGlyph[] = [];
    private bbox = new BBOX();

    private program: WebGLProgram;
    private reflMap: WebGLTexture;

    private a_Position: number;
    private a_Normal: number;
    private u_ClipFromView: WebGLUniformLocation;
    private u_ViewFromLocal: WebGLUniformLocation;
    private u_ViewFromLocalNormal: WebGLUniformLocation;

    private scaleXY: number = 0.0001;
    private scaleZ: number = 0.03;

    private worldFromLocal = mat4.create();

    // Public API
    public viewFromWorld = mat4.create();

    constructor(gl: WebGLRenderingContext, private fontData: FontData, reflMap: TexImageSource) {
        this.program = createTextProgram(gl);
        this.a_Position = gl.getAttribLocation(this.program, `a_Position`);
        this.a_Normal = gl.getAttribLocation(this.program, `a_Normal`);
        this.u_ClipFromView = gl.getUniformLocation(this.program, `u_ClipFromView`);
        this.u_ViewFromLocal = gl.getUniformLocation(this.program, `u_ViewFromLocal`);
        this.u_ViewFromLocalNormal = gl.getUniformLocation(this.program, `u_ViewFromLocalNormal`);

        this.reflMap = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, this.reflMap);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, reflMap);
    }

    public layout(gl: WebGLRenderingContext, text: string): void {
        bboxReset(this.bbox);

        const font = this.fontData.font;
        const metricsScale = 1 / font.unitsPerEm * FONT_SIZE;

        font.forEachGlyph(text, 0, 0, FONT_SIZE, undefined, (glyph, x, y) => {
            const glyphIndex = (glyph as any).index;
            this.fontData.ensureGlyph(gl, glyphIndex);
            this.placedGlyphs.push({ x, y, glyphIndex });
            const metrics = glyph.getMetrics();
            const xMin = x + metrics.xMin * metricsScale;
            const xMax = x + metrics.xMax * metricsScale;
            const yMin = y + metrics.yMin * metricsScale;
            const yMax = y + metrics.yMax * metricsScale;
            bboxUnionPoint(this.bbox, xMin, yMin);
            bboxUnionPoint(this.bbox, xMax, yMin);
            bboxUnionPoint(this.bbox, xMin, yMax);
            bboxUnionPoint(this.bbox, xMax, yMax);
        });

        // Sort by glyph index.
        this.placedGlyphs.sort((a, b) => a.glyphIndex - b.glyphIndex);
    }

    public render(gl: WebGLRenderingContext): void {
        gl.useProgram(this.program);
        gl.enable(gl.DEPTH_TEST);
        gl.depthMask(true);

        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.reflMap);

        const viewportWidth = gl.canvas.width, viewportHeight = gl.canvas.height;
        gl.viewport(0, 0, viewportWidth, viewportHeight);
        mat4.perspective(scratchMatrix, Math.PI / 4, viewportWidth / viewportHeight, 0.2, 256);
        gl.uniformMatrix4fv(this.u_ClipFromView, false, scratchMatrix);

        mat4.identity(this.worldFromLocal);
        this.worldFromLocal[0] = this.scaleXY;
        this.worldFromLocal[5] = this.scaleXY;
        this.worldFromLocal[10] = this.scaleZ;
        // Y flip because GL.
        this.worldFromLocal[5] *= -1;

        mat4.mul(scratchMatrix, this.viewFromWorld, this.worldFromLocal);

        // Normal matrix.
        computeNormalMatrix(scratchMatrix, scratchMatrix);
        gl.uniformMatrix4fv(this.u_ViewFromLocalNormal, false, scratchMatrix);

        const offsX = (this.bbox.maxX - this.bbox.minX) * -0.5;
        const offsY = (this.bbox.maxY - this.bbox.minY) * 0.5;

        // Render text itself.
        let glyphIndex = -1;
        let indexCount = -1;
        for (let i = 0; i < this.placedGlyphs.length; i++) {
            const g = this.placedGlyphs[i];

            if (glyphIndex !== g.glyphIndex) {
                glyphIndex = g.glyphIndex;

                const glyphData = this.fontData.getGlyph(glyphIndex);
                if (glyphData === null) {
                    // No data; probably from whitespace.
                    indexCount = 0;
                    continue;
                }

                gl.bindBuffer(gl.ARRAY_BUFFER, glyphData.vertexBuffer);
                gl.vertexAttribPointer(this.a_Position, 3, gl.FLOAT, false, 6*4, 0);
                gl.vertexAttribPointer(this.a_Normal, 3, gl.FLOAT, false, 6*4, 3*4);
                gl.enableVertexAttribArray(this.a_Position);
                gl.enableVertexAttribArray(this.a_Normal);
                gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, glyphData.indexBuffer);
                indexCount = glyphData.indexCount;
            }

            if (indexCount === 0)
                continue;

            mat4.identity(scratchMatrix);
            scratchMatrix[12] += offsX + g.x;
            scratchMatrix[13] += offsY + g.y;
            mat4.mul(scratchMatrix, this.worldFromLocal, scratchMatrix);
            mat4.mul(scratchMatrix, this.viewFromWorld, scratchMatrix);

            gl.uniformMatrix4fv(this.u_ViewFromLocal, false, scratchMatrix);

            gl.drawElements(gl.TRIANGLES, indexCount, gl.UNSIGNED_SHORT, 0);
        }
    }

    public destroy(gl: WebGLRenderingContext): void {
        gl.deleteProgram(this.program);
        gl.deleteTexture(this.reflMap);
        this.fontData.destroy(gl);
    }
}

async function decodeImage(data: Uint8Array): Promise<TexImageSource> {
    const img = document.createElement('img');
    const blob = new Blob([data]);
    const url = window.URL.createObjectURL(blob);
    img.src = url;
    await img.decode();
    return img;
}

class Main {
    private textRenderer: TextRenderer | null = null;
    private canvas: HTMLCanvasElement;
    private gl: WebGLRenderingContext;

    // Animation stuffs.
    private phase: number = 0;
    private x: number = 0;
    private y: number = 0;
    private xa: number = 0.001;
    private ya: number = 0.001;
    private xm: number = 0.3;
    private ym: number = 0.25;
    private zDist: number = -0.7;

    public string: string = "WELCOEM TO MY WEBSITE";

    constructor() {
        this.canvas = document.createElement('canvas');
        this.canvas.style.position = 'absolute';
        this.canvas.style.top = '0';
        this.canvas.style.left = '0';
        this.canvas.style.pointerEvents = 'none';
        this.gl = this.canvas.getContext('webgl');
        document.body.appendChild(this.canvas);

        window.onresize = this.onresize.bind(this);
        this.onresize();

        window.oncontextmenu = (e) => {
            e.preventDefault();
        };
        
        this.initAnimation();
        this.load();
        this.update();
    }

    private initAnimation(): void {
        this.phase = Math.floor(Math.random() * 10000);

        for (let i = 0; i < this.phase; i++)
            this.updateBounce();
    }

    private updateBounce(): void {
        this.x += this.xa;
        this.y += this.ya;
    
        if (Math.abs(this.x) > this.xm)
            this.xa *= -1;
        if (Math.abs(this.y) > this.ym)
            this.ya *= -1;
    }

    private onresize(): void {
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
    }

    public async load() {
        function Buffer(b64: string): Uint8Array {
            return Uint8Array.from(window.atob(b64), function(c) { return c.charCodeAt(0); });
        }

        const fontBuffer = readFileSync('./COMICBD.ttf') as Uint8Array;
        const reflMapBuffer = readFileSync('./ReflMap.jpg') as Uint8Array;
        const fontData = new FontData(opentype.parse(fontBuffer.buffer));

        const reflMap = await decodeImage(reflMapBuffer);
        this.textRenderer = new TextRenderer(this.gl, fontData, reflMap);

        this.relayoutText();
    }

    private relayoutText(): void {
        this.textRenderer.layout(this.gl, this.string);
    }

    private update = () => {
        const gl = this.gl;

        gl.clearColor(0, 0, 0, 1);
        gl.clear(gl.COLOR_BUFFER_BIT);
    
        const textRenderer = this.textRenderer;
        if (textRenderer !== null) {
            const time = (this.phase + window.performance.now()) / 2000;
    
            mat4.identity(textRenderer.viewFromWorld);
            mat4.rotateY(textRenderer.viewFromWorld, textRenderer.viewFromWorld, time);
            this.updateBounce();
    
            textRenderer.viewFromWorld[12] = this.x;
            textRenderer.viewFromWorld[13] = this.y;
            textRenderer.viewFromWorld[14] = this.zDist;
            textRenderer.render(gl);
        }
    
        requestAnimationFrame(this.update);
    };
}

declare global {
    interface Window {
        main: Main;
    }
}

window.main = new Main();
