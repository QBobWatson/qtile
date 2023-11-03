
import ctypes
import os

from contextlib import suppress
from math import copysign
from typing import Any

from OpenGL import GLES2 as GL
from OpenGL import EGL, platform

from pywayland import lib as pwlib
from pywayland.protocol.wayland import WlOutput

from wlroots.util.box import Box
from wlroots.util.region import PixmanRegion32
from wlroots import ffi, lib
from wlroots.wlr_types import SceneNode, SceneNodeType, Matrix
from wlroots.wlr_types.scene import SceneBuffer, SceneRect
from wlroots.wlr_types import Texture
from wlroots.wlr_types import Output as wlrOutput
from wlroots.renderer import Renderer as wlrRenderer

from libqtile.backend.wayland.output import Output
from libqtile.backend.wayland.window import Window, WindowType, _rgb as rgb

# from libqtile.log_utils import logger


def _scene_buffer_get_texture(scene_buffer: SceneBuffer, renderer: wlrRenderer) -> Texture:
    """Get the texture associated with a SceneBuffer.

    This is copied directly from the function of the same name in wlr_scene.c.
    It accesses the private field wlr_scene_buffer.texture.
    """
    buffer = scene_buffer.buffer
    client_buffer = lib.wlr_client_buffer_get(buffer._ptr)
    if client_buffer != ffi.NULL:
        return Texture(client_buffer.texture)
    texture = scene_buffer.texture
    if texture is not None:
        return texture
    texture = Texture.from_buffer(renderer, buffer)
    ffi.gc(texture._ptr, None)
    scene_buffer.texture = texture
    return texture


class Program:
    """A GLSL shader program."""

    def __init__(self, vert_src: str, frag_src: str,
                 defines: list[tuple[str, Any] | str] | None = None) -> None:
        if defines:
            defines_strs = []
            for item in defines:
                if isinstance(item, tuple):
                    key, val = item
                    defines_strs.append(f"#define {key} {val}\n")
                else:
                    defines_strs.append(f"#define {item}\n")
            defines_str = ''.join(defines_strs)
            vert_src = defines_str + vert_src
            frag_src = defines_str + frag_src

        vert = self._compile_shader(GL.GL_VERTEX_SHADER, vert_src)
        try:
            frag = self._compile_shader(GL.GL_FRAGMENT_SHADER, frag_src)
        except RuntimeError:
            GL.glDeleteShader(vert)
            raise
        prog = GL.glCreateProgram()
        GL.glAttachShader(prog, vert)
        GL.glAttachShader(prog, frag)
        GL.glLinkProgram(prog)
        GL.glDeleteShader(vert)
        GL.glDeleteShader(frag)

        ok = GL.glGetProgramiv(prog, GL.GL_LINK_STATUS)
        if ok == GL.GL_FALSE:
            raise RuntimeError(
                "Program link failure: %s" % GL.glGetProgramInfoLog(prog).decode()
            )

        self.program = prog

        self.uniforms = {}
        count = GL.glGetProgramiv(prog, GL.GL_ACTIVE_UNIFORMS)
        for i in range(count):
            name, _, _ = GL.glGetActiveUniform(prog, i)
            name = ''.join(chr(x) for x in name if x != 0)
            self.uniforms[name] = GL.glGetUniformLocation(prog, name)

        self.attributes = {}
        count = GL.glGetProgramiv(prog, GL.GL_ACTIVE_ATTRIBUTES)
        for i in range(count):
            name, _, _ = GL.glGetActiveAttrib(prog, i)
            name = ''.join(chr(x) for x in name if x != 0)
            self.attributes[name] = GL.glGetAttribLocation(prog, name)

    def _compile_shader(self, type: int, src: str) -> int:
        shader = GL.glCreateShader(type)
        GL.glShaderSource(shader, [src])
        GL.glCompileShader(shader)
        ok = GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS)
        if ok == GL.GL_FALSE:
            raise RuntimeError(
                "Shader compile failure: %s" % GL.glGetShaderInfoLog(shader).decode()
            )
        return shader


class _RenderOutput:
    """Helper class containing output-specific render state.

    This is computed once per frame.
    """

    def __init__(self, renderer: Any, output: Output, damage: PixmanRegion32):
        self.renderer = renderer
        self.output = output
        """The region we have to draw this frame."""
        self.damage = damage
        self.wlr_output = output.wlr_output
        self.scene_output = output.scene_output
        width, height = self.wlr_output.effective_resolution()
        # output_box is in layout coordinates
        self.output_box = Box(
            self.scene_output.x, self.scene_output.y, width, height)
        self.scaled_width, self.scaled_height = \
            self.wlr_output.transformed_resolution()
        self.output_inv_transform = \
            wlrOutput.transform_invert(self.wlr_output.transform)
        self.projection = self._compute_projection_matrix()

    @staticmethod
    def _scale_length(length: int, offset: int, scale: float) -> int:
        return round((offset + length) * scale) - round(offset * scale)

    @staticmethod
    def _scale_box(box: Box, scale: float) -> None:
        box.width = _RenderOutput._scale_length(box.width, box.x, scale)
        box.height = _RenderOutput._scale_length(box.height, box.y, scale)
        box.x = round(box.x * scale)
        box.y = round(box.y * scale)

    def _compute_projection_matrix(self) -> Matrix:
        """Compute the projection matrix for this output.

        Copied from wlr_matrix.c (not a public API for some reason).
        """
        trans = Matrix.identity()
        trans.transform(WlOutput.transform.flipped_180)
        x = 2 / self.wlr_output.width
        y = 2 / self.wlr_output.height
        proj = Matrix.identity()
        t = trans._ptr
        mat = proj._ptr
        # Rotation + reflection
        mat[0] = x * t[0]
        mat[1] = x * t[1]
        mat[3] = -y * t[3]
        mat[4] = -y * t[4]
        # Translation
        mat[2] = -copysign(1, mat[0] + mat[1])
        mat[5] = -copysign(1, mat[3] + mat[4])
        return proj

    def box_layout_to_damage_coords(self, box: Box) -> None:
        """Change a box from layout coordinates to damage coordinates."""
        box.x -= self.scene_output.x
        box.y -= self.scene_output.y
        self._scale_box(box, self.wlr_output.scale)

    def box_damage_to_output_coords(self, box: Box) -> None:
        """Change a box from damage coordinates to output coordinates."""
        box.transform(
            box,
            self.output_inv_transform,
            self.scaled_width,
            self.scaled_height
        )

    def region_layout_to_damage_coords(
            self,
            region: PixmanRegion32,
            round_up: bool = False,
    ) -> None:
        """Change a region from layout coordinates to damage coordinates."""
        region.translate(-self.scene_output.x, -self.scene_output.y)
        region.scale(region, self.wlr_output.scale)
        if round_up and not self.wlr_output.scale.is_integer():
            region.expand(region, 1)

    def region_damage_to_output_coords(self, region: PixmanRegion32) -> None:
        """Change a region from damage coordinates to output coordinates."""
        region.transform(
            region,
            self.output_inv_transform,
            self.scaled_width,
            self.scaled_height
        )

    def contains_box_layout_coords(self, box: Box) -> bool:
        """Check if a box (in layout coordinates) intersects this output."""
        return Box().intersection(box, self.output_box)

    def contains_region_layout_coords(self, region: PixmanRegion32) -> bool:
        """Check if a region (in layout coordinates) intersects this output."""
        with PixmanRegion32() as intersection:
            # node.visible is in layout coordinates
            intersection.intersect_rect(
                region,
                self.output_box.x,
                self.output_box.y,
                self.output_box.width,
                self.output_box.height
            )
            return intersection.not_empty()

    def compute_render_region(
            self,
            extents: Box,
            region: PixmanRegion32
    ) -> PixmanRegion32 | None:
        """Check if any part of `region` should be redrawn on this output.

        The `extents` parameter should be equivalent to the region's extents.
        If the extents box intersects this output and the region intersects the
        current frame damage, the intersection is returned; otherwise None is
        returned.

        All boxes are in layout coordinates.
        """
        if not self.contains_box_layout_coords(extents) or \
           not self.contains_region_layout_coords(region):
            return
        render_region = PixmanRegion32()
        render_region.init()
        render_region.copy_from(region)
        self.region_layout_to_damage_coords(render_region, True)
        render_region.intersect(render_region, self.damage)
        if not render_region.not_empty():
            render_region.fini()
            return
        return render_region


class _RenderNode:
    """Helper class to pass around context for rendering one node."""
    def __init__(
            self,
            render_output: _RenderOutput,
            node: SceneNode,
            x: int, y: int,
            win: WindowType | None = None,
            alpha: float = 1.0,
            corner: int = 0
    ) -> None:
        self.render_output = render_output
        """The node we're rendering."""
        self.node = node
        """The window this node belongs to, if any."""
        self.win = win
        if win is None and node.data is not None and node.type == SceneNodeType.TREE:
            self.win = node.data
        self.alpha = alpha
        self.corner = corner
        width, height = node.size
        """The extents of the node, in layout coordinates."""
        self.node_box = Box(x, y, width, height)
        """The extents of the node, in damage coordinates."""
        self.dst_box = Box(x, y, width, height)
        render_output.box_layout_to_damage_coords(self.dst_box)
        # for RECT and BUFFER
        self._render_region = None
        # for BUFFER
        self._texture = None
        self._texture_attribs = None

    def finalize(self):
        if self._render_region:
            self._render_region.fini()

    @property
    def render_region(self) -> PixmanRegion32 | None:
        """Compute the region of this node that we have to render."""
        if self._render_region is None:
            self._render_region = self._compute_render_region()
        if self._render_region is False:
            return
        return self._render_region

    @property
    def texture(self) -> Texture | None:
        """Extract the texture underlying a BUFFER node."""
        if self._texture is None:
            self._texture, self._texture_attribs = self._scene_buffer_get_texture()
        if self._texture is False:
            return
        return self._texture

    @property
    def texture_attribs(self) -> Texture | None:
        """The GL texture data underlying `texture`."""
        if self._texture_attribs is None:
            self._texture, self._texture_attribs = self._scene_buffer_get_texture()
        if self._texture_attribs is False:
            return
        return self._texture_attribs

    def child(self, node: SceneNode) -> Any:
        """Wrap a child node with a _RenderNode."""
        return _RenderNode(
            self.render_output,
            node,
            self.node_box.x + node.x,
            self.node_box.y + node.y,
            self.win,
        )

    def _compute_render_region(self) -> PixmanRegion32 | bool:
        region = self.render_output.compute_render_region(
            self.node_box, self.node.visible)
        return region or False

    def _scene_buffer_get_texture(self) -> tuple[Texture | bool, Any]:
        texture = _scene_buffer_get_texture(
            self.node.as_buffer, self.render_output.renderer.wlr_renderer)
        if not texture:
            return False, False
        assert lib.wlr_texture_is_gles2(texture._ptr)
        attribs = ffi.new("struct wlr_gles2_texture_attribs *")
        lib.wlr_gles2_texture_get_attribs(texture._ptr, attribs)
        return texture, attribs


class Renderer:
    """Custom renderer class.

    This is responsible for rendering the scene graph to each output.
    """

    quad_verts = [
        1, 0,  # top right
        0, 0,  # top left
        1, 1,  # bottom right
        0, 1,  # bottom left
    ]

    def __init__(self, core):
        self.core = core
        self.wlr_renderer = core.renderer
        # HACK!  This is the only way to convince the scene graph not to
        # occlude the area under rounded corners.
        self.core.scene._ptr.calculate_visibility = False

        # Make context current using raw EGL calls...
        wlr_egl = lib.wlr_gles2_renderer_get_egl(self.wlr_renderer._ptr)
        display = lib.wlr_egl_get_display(wlr_egl)
        context = lib.wlr_egl_get_context(wlr_egl)
        # magic to cast from ffi type to ctype
        display = ctypes.cast(int(ffi.cast("intptr_t", display)), EGL.EGLDisplay)
        context = ctypes.cast(int(ffi.cast("intptr_t", context)), EGL.EGLContext)
        if not EGL.eglMakeCurrent(
                display, EGL.EGL_NO_SURFACE, EGL.EGL_NO_SURFACE, context):
            raise RuntimeError("Cannot make EGL context current")

        self.init_shaders()

        EGL.eglMakeCurrent(
            display, EGL.EGL_NO_SURFACE, EGL.EGL_NO_SURFACE, EGL.EGL_NO_CONTEXT)

    def init_shaders(self):
        """Compile the shader programs we'll use."""
        self.shaders = {}
        glsl_dir = os.path.join(os.path.dirname(__file__), 'glsl')
        for fname in os.listdir(glsl_dir):
            with open(os.path.join(glsl_dir, fname)) as f:
                self.shaders[fname] = f.read()

        self.programs = {}
        self.programs['quad'] = Program(
            self.shaders['common.vert'], self.shaders['quad.frag'],
        )
        self.programs['border'] = Program(
            self.shaders['common.vert'], self.shaders['border.frag'],
        )
        self.programs['tex_rgba'] = Program(
            self.shaders['common.vert'], self.shaders['tex.frag'],
            defines=['TEX_RGBA']
        )
        self.programs['tex_rgbx'] = Program(
            self.shaders['common.vert'], self.shaders['tex.frag'],
            defines=['TEX_RGBX']
        )
        # seems to be broken...
        # exts = GL.glGetString(GL.GL_VENDOR)
        gles2 = platform.PLATFORM.GLES2
        gles2.glGetString.restype = ctypes.c_char_p
        exts = gles2.glGetString(GL.GL_EXTENSIONS).decode().split()
        if 'GL_OES_EGL_image_external' in exts:
            self.programs['tex_ext'] = Program(
                self.shaders['common.vert'], self.shaders['tex.frag'],
                defines=['TEX_EXT']
            )

    def finalize(self):
        pass

    def create_borders(self, win: Window) -> None:
        """Replacement for Window.paint_borders().

        If a custom renderer is installed, Window.paint_borders() calls this
        method to create the nodes used for its borders.  The border nodes are
        [SceneRect] instances which are (probably) not painted using
        self.render_rect(), but are needed so that the scene graph will still
        do damage tracking on the borders.
        """
        tree = win.container
        borders = [node.as_rect for node in tree.children
                   if node.type == SceneNodeType.RECT]
        # We use 8 border rects (one for each corner, one for each side).
        # We can re-use the old borders if there are 8 of them.
        if len(borders) != 8:
            # Clear the old borders
            for rect in borders:
                rect.node.destroy()
            borders = []
        corner_w = corner_h = getattr(win, 'bordercorner', 0)
        if corner_w * 2 > win.width:
            corner_w = win.width // 2
        if corner_h * 2 > win.height:
            corner_h = win.height // 2
        bw = win.borderwidth

        # x1, y1, x2, y2, relative to window top-left
        geometries = (
            (-bw, -bw, corner_w, corner_h),  # top left
            (corner_w, -bw, win.width - corner_w, 0),  # top
            (win.width - corner_w, -bw, win.width + bw, corner_h),  # top right
            (-bw, corner_h, 0, win.height - corner_h),  # left
            (win.width, corner_h, win.width + bw, win.height - corner_h),  # right
            (-bw, win.height - corner_h, corner_w, win.height + bw),  # bottom left
            (corner_w, win.height, win.width - corner_w, win.height + bw),  # bottom
            (win.width - corner_w, win.height - corner_h, win.width + bw, win.height + bw),  # bottom right
        )

        color = rgb(win.bordercolor[0])
        if borders:
            for (x1, y1, x2, y2), rect in zip(geometries, borders):
                rect.set_color(color)
                rect.set_size(x2 - x1, y2 - y1)
                rect.node.set_position(x1 + bw, y1 + bw)
        else:
            for (x1, y1, x2, y2) in geometries:
                rect = SceneRect.create(tree, x2 - x1, y2 - y1, color)
                rect.node.set_position(x1 + bw, y1 + bw)
        if win.tree:
            win.tree.node.raise_to_top()

    def scissor_in_damage_coords(self, render_output: _RenderOutput, box: Box) -> None:
        render_output.box_damage_to_output_coords(box)
        self.wlr_renderer.scissor(box)

    def render_rect(self, render_node: _RenderNode) -> None:
        """Render a SceneRect as a solid-colored box."""
        box = render_node.dst_box
        if box.width == 0 or box.height == 0:
            return

        render_output = render_node.render_output
        matrix = Matrix.project_box(
            box, WlOutput.transform.normal, 0.0, render_output.wlr_output.transform_matrix)
        gl_matrix = (render_output.projection @ matrix).transpose()

        color = render_node.node.as_rect.color
        if color[3] == 1.0:
            GL.glDisable(GL.GL_BLEND)
        else:
            GL.glEnable(GL.GL_BLEND)

        program = self.programs['quad']
        GL.glUseProgram(program.program)
        GL.glUniformMatrix3fv(
            program.uniforms['proj'], 1, GL.GL_FALSE, list(gl_matrix._ptr))
        GL.glUniform4f(program.uniforms['color'], *color)
        GL.glVertexAttribPointer(
            program.attributes['pos'], 2,
            GL.GL_FLOAT, GL.GL_FALSE, 0, self.quad_verts)
        GL.glEnableVertexAttribArray(program.attributes['pos'])

        for rect in render_node.render_region.rectangles_as_boxes():
            self.scissor_in_damage_coords(render_output, rect)
            GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)

        GL.glDisableVertexAttribArray(program.attributes['pos'])

    def render_buffer_setup(self, render_node: _RenderNode, program: Program) -> None:
        """Common code to set up a program that renders a buffer."""
        scene_buffer = render_node.node.as_buffer
        transform = wlrOutput.transform_invert(scene_buffer.transform)
        matrix = Matrix.project_box(
            render_node.dst_box, transform, 0.0,
            render_node.render_output.wlr_output.transform_matrix)
        gl_matrix = (render_node.render_output.projection @ matrix).transpose()

        attribs = render_node.texture_attribs
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(attribs.target, attribs.tex)
        GL.glTexParameteri(attribs.target, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glUseProgram(program.program)
        with suppress(KeyError):
            GL.glUniformMatrix3fv(
                program.uniforms['proj'], 1, GL.GL_FALSE, list(gl_matrix._ptr))
        with suppress(KeyError):
            GL.glUniform1i(program.uniforms['tex'], 0)
        with suppress(KeyError):
            GL.glUniform2f(program.uniforms['boxSize'],
                           render_node.dst_box.width, render_node.dst_box.height)
        with suppress(KeyError):
            GL.glUniform1f(program.uniforms['alpha'], render_node.alpha)
        with suppress(KeyError):
            GL.glUniform1f(program.uniforms['corner'],
                           render_node.corner *
                           render_node.render_output.wlr_output.scale)

        texture = render_node.texture
        src_box = ffi.addressof(scene_buffer._ptr.src_box)
        if src_box.width <= 0 or src_box.height <= 0:
            src_box = Box(0, 0, texture.width, texture.height)
        x1 = src_box.x / texture.width
        y1 = src_box.y / texture.height
        x2 = (src_box.x + src_box.width) / texture.width
        y2 = (src_box.y + src_box.height) / texture.height
        texcoord = [
            x2, y1,  # top right
            x1, y1,  # top left
            x2, y2,  # bottom right
            x1, y2,  # bottom left
        ]

        with suppress(KeyError):
            GL.glVertexAttribPointer(
                program.attributes['pos'], 2,
                GL.GL_FLOAT, GL.GL_FALSE, 0, self.quad_verts)
            GL.glEnableVertexAttribArray(program.attributes['pos'])
        with suppress(KeyError):
            GL.glVertexAttribPointer(
                program.attributes['texcoord'], 2,
                GL.GL_FLOAT, GL.GL_FALSE, 0, texcoord)
            GL.glEnableVertexAttribArray(program.attributes['texcoord'])

        return attribs

    def render_buffer_finish(self, render_node: _RenderNode, program: Program) -> None:
        """Common code to reset state after rendering a buffer."""
        with suppress(KeyError):
            GL.glDisableVertexAttribArray(program.attributes['pos'])
        with suppress(KeyError):
            GL.glDisableVertexAttribArray(program.attributes['texcoord'])
        GL.glBindTexture(render_node.texture_attribs.target, 0)

        pwlib.wl_signal_emit(
            ffi.addressof(render_node.node.as_buffer._ptr.events.output_present),
            render_node.render_output.scene_output._ptr
        )

    def render_buffer(self, render_node: _RenderNode) -> None:
        """Render a buffer as a textured rectangle."""
        texture = render_node.texture
        if texture is None:
            return

        attribs = render_node.texture_attribs
        if attribs.target == GL.GL_TEXTURE_2D:
            if attribs.has_alpha or render_node.alpha < 1 or render_node.corner > 0:
                program = self.programs['tex_rgba']
            else:
                program = self.programs['tex_rgbx']
        elif attribs.target == GL.GL_TEXTURE_EXTERNAL_OES:
            program = self.programs['tex_ext']
        else:
            assert False

        self.render_buffer_setup(render_node, program)

        if attribs.has_alpha or render_node.alpha < 1 or render_node.corner > 0:
            GL.glEnable(GL.GL_BLEND)
        else:
            GL.glDisable(GL.GL_BLEND)

        for box in render_node.render_region.rectangles_as_boxes():
            self.scissor_in_damage_coords(render_node.render_output, box)
            GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)

        self.render_buffer_finish(render_node, program)

    def render_main_window_surface(self, render_node: _RenderNode) -> None:
        """Render a buffer that [render_window_buffer] has identified as the
        primary window content.

        For now, we just round the corners.
        """
        render_node.corner = getattr(render_node.win, 'bordercorner', 0)
        self.render_buffer(render_node)

    def render_border_setup(
            self,
            render_node: _RenderNode,
            program: Program,
            borders: list[SceneNode] | None = None
    ) -> PixmanRegion32 | None:
        """Common code to set up the border rendering program."""
        win = render_node.win
        bw = getattr(win, "borderwidth", 0)
        if bw == 0:
            return

        # Compute the render region
        extents = Box(win.x, win.y, win.width + bw * 2, win.height + bw * 2)
        render_output = render_node.render_output
        borders = borders or [
            node.as_rect for node in render_node.node.as_tree.children
            if node.type == SceneNodeType.RECT
        ]
        with PixmanRegion32() as region:
            for rect in borders:
                region.union_rect(
                    region, rect.node.x, rect.node.y, rect.width, rect.height)
            region.translate(render_node.node_box.x, render_node.node_box.y)
            render_region = render_output.compute_render_region(extents, region)
        if not render_region:
            return

        render_output.box_layout_to_damage_coords(extents)
        matrix = Matrix.project_box(
            extents, WlOutput.transform.normal, 0.0,
            render_output.wlr_output.transform_matrix)
        gl_matrix = (render_output.projection @ matrix).transpose()

        corner = getattr(render_node.win, 'bordercorner', 0)
        color = rgb(win.bordercolor[0])
        if color[3] < 1.0 or corner > 0:
            GL.glEnable(GL.GL_BLEND)
        else:
            GL.glDisable(GL.GL_BLEND)

        scale = render_node.render_output.wlr_output.scale
        GL.glUseProgram(program.program)
        with suppress(KeyError):
            GL.glUniformMatrix3fv(
                program.uniforms['proj'], 1, GL.GL_FALSE, list(gl_matrix._ptr))
        with suppress(KeyError):
            GL.glUniform4f(program.uniforms['color'], *color)
        with suppress(KeyError):
            GL.glUniform1f(program.uniforms['corner'], corner * scale)
        with suppress(KeyError):
            GL.glUniform1f(program.uniforms['borderWidth'], bw * scale)
        with suppress(KeyError):
            GL.glUniform2f(program.uniforms['boxSize'],
                           extents.width, extents.height)
        with suppress(KeyError):
            GL.glVertexAttribPointer(
                program.attributes['pos'], 2,
                GL.GL_FLOAT, GL.GL_FALSE, 0, self.quad_verts)
        with suppress(KeyError):
            GL.glEnableVertexAttribArray(program.attributes['pos'])

        return render_region

    def render_window_borders(self, render_node: _RenderNode) -> None:
        """Render window borders.

        The border rects are just placeholders for damage tracking.  This is
        where the borders actually are rendered.
        """
        render_output = render_node.render_output
        program = self.programs['border']
        render_region = self.render_border_setup(render_node, program)
        if not render_region:
            return

        for rect in render_region.rectangles_as_boxes():
            self.scissor_in_damage_coords(render_output, rect)
            GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)

        GL.glDisableVertexAttribArray(program.attributes['pos'])

        render_region.fini()

    def render_window_buffer(self, render_node: _RenderNode) -> None:
        """Decide how to render a buffer descended from a window.

        How should we render this buffer?  It's not always clear what is the
        purpose of the various surfaces a client produces, and how they should
        be decorated.  Windows trees have surfaces and subsurfaces, popups,
        nested popups, etc.  xdg clients are at least nice enough to inform us
        of the roles of their top-level surfaces, but the subsurfaces could be
        anything...
        """
        # Is this the main surface for the window?  It's hard to tell for sure,
        # so we use this criterion: if the buffer size is the same as the
        # window size, then it's the main surface.
        if render_node.node_box.width == render_node.win.width and \
           render_node.node_box.height == render_node.win.height:
            self.render_main_window_surface(render_node)
        else:
            self.render_buffer(render_node)

        # TODO: popups, other roles, ...

    def render_window_tree(self, render_node: _RenderNode) -> None:
        """Render a SceneTree corresponding to a window.

        The main point of this method is to use [render_window_borders()]
        instead of rendering the border rects with [render_rect()].
        """
        self.render_window_borders(render_node)
        for child in render_node.node.as_tree.children:
            # Don't render borders again.
            if child.type == SceneNodeType.RECT:
                continue
            self.render_node(render_node.child(child))

    def render_node(self, render_node: _RenderNode) -> None:
        """Generic method for rendering a node.

        This dispatches the rendering to one of the render_* calls above,
        depending on what kind of node we're dealing with.
        """
        node = render_node.node
        if not node.enabled:
            return

        if node.data is not None and node.type == SceneNodeType.TREE:
            self.render_window_tree(render_node)
            return

        if node.type == SceneNodeType.TREE:
            tree = node.as_tree
            for child in tree.children:
                self.render_node(render_node.child(child))
            return

        # node.type is RECT or BUFFER
        # First decide if we have to draw anything.
        if node.invisible or not render_node.render_region:
            return

        # Ok, we have to draw something.
        if node.type == SceneNodeType.RECT:
            self.render_rect(render_node)
        elif node.type == SceneNodeType.BUFFER:
            if render_node.win is not None:
                self.render_window_buffer(render_node)
            else:
                self.render_buffer(render_node)

        render_node.finalize()

    def render(self, output: Output) -> None:
        """Render the scene graph to [output]."""
        wlr_output = output.wlr_output
        scene_output = output.scene_output
        damage_ring = ffi.addressof(scene_output._ptr.damage_ring)
        damage_current = PixmanRegion32(ffi.addressof(damage_ring.current))

        if not wlr_output.needs_frame and not damage_current.not_empty():
            return

        buffer_age = wlr_output.attach_render()

        with PixmanRegion32() as damage:
            # damage is in output coordinates, after scaling but before rotating
            lib.wlr_damage_ring_get_buffer_damage(
                damage_ring, buffer_age, damage._ptr)
            render_output = _RenderOutput(self, output, damage)

            wlr_renderer = self.wlr_renderer
            wlr_renderer.begin(wlr_output.width, wlr_output.height)

            for box in damage.rectangles_as_boxes():
                self.scissor_in_damage_coords(render_output, box)
                wlr_renderer.clear([0, 0, 0, 1])

            # print_scene_tree(tree)
            node = self.core.scene.tree.node
            self.render_node(_RenderNode(render_output, node, node.x, node.y))

            wlr_renderer.scissor(None)
            wlr_output.render_software_cursors(damage)

            wlr_renderer.end()

        with PixmanRegion32() as frame_damage:
            frame_damage.copy_from(damage_current)
            render_output.region_damage_to_output_coords(frame_damage)
            wlr_output.set_damage(frame_damage)

        success = lib.wlr_output_commit(wlr_output._ptr)
        if success:
            lib.wlr_damage_ring_rotate(damage_ring)
        return success
