uniform mat3 proj;
attribute vec2 pos;
attribute vec2 texcoord;
varying vec2 v_pos;
varying vec2 v_texcoord;

void main() {
  gl_Position = vec4(proj * vec3(pos, 1.0), 1.0);
  v_pos = pos;
  v_texcoord = texcoord;
}
