#ifdef TEX_EXT
#extension GL_OES_EGL_image_external : require
#endif
#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif

#if defined(TEX_RGBX) || defined(TEX_RGBA)
uniform sampler2D tex;
#elif defined(TEX_EXT)
uniform samplerExternalOES tex;
#endif

varying vec2 v_pos;
varying vec2 v_texcoord;
uniform float alpha;
uniform float corner;
uniform vec2 boxSize;


void main() {
  float len;
  if(corner > 0.0) {
    vec2 boxPos = (v_pos - vec2(0.5)) * boxSize;
    len = length(max(abs(boxPos) - boxSize / 2.0 + vec2(corner), vec2(0.0)));
    if(len > corner + 0.5) discard;
  }
#if defined(TEX_RGBX)
  gl_FragColor = vec4(texture2D(tex, v_texcoord).rgb, 1.0) * alpha;
#elif defined(TEX_RGBA) || defined(TEX_EXT)
  gl_FragColor = texture2D(tex, v_texcoord) * alpha;
#endif
  if(corner > 0.0)
    gl_FragColor *= min(corner - len + 0.5, 1.0);
}
