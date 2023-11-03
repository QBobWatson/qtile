#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif

varying vec2 v_pos;
uniform vec4 color;
uniform float corner;
uniform float borderWidth;
uniform vec2 boxSize;


void main() {
  float len;
  if(corner > 0.0) {
    vec2 boxPos = (v_pos - vec2(0.5)) * boxSize;
    len = length(max(abs(boxPos) - boxSize / 2.0 + vec2(corner + borderWidth), vec2(0.0)));
    if(len > corner + borderWidth + 1.0) discard;
    if(len < corner - 0.5) discard;
  }
  gl_FragColor = color;
  if(corner > 0.0)
    gl_FragColor *= min(min(corner + borderWidth + 1.0 - len, len - corner + 0.5), 1.0);
}
