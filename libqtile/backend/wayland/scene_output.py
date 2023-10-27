
# from OpenGL import GLES2 as GL
from pywayland import lib as pwlib

from wlroots.util.box import Box
from wlroots.util.region import PixmanRegion32
from wlroots import ffi, lib
from wlroots.wlr_types import SceneNode, SceneNodeType, Matrix
from wlroots.wlr_types.scene import SceneBuffer, SceneRect
from wlroots.wlr_types import Texture
from wlroots.wlr_types import Scene as wlrScene
from wlroots.wlr_types import Output as wlrOutput
from wlroots.wlr_types import SceneOutput as wlrSceneOutput
from wlroots.renderer import Renderer

from typing import Iterator

# from libqtile.log_utils import logger


def _scene_nodes_in_box(node: SceneNode, box: Box, lx: int, ly: int) -> Iterator[SceneNode]:
    if not node.enabled:
        return

    if node.type == SceneNodeType.TREE:
        tree = node.as_tree
        children = list(tree.children)
        children.reverse()
        for child in children:
            yield from _scene_nodes_in_box(child, box, lx + child.x, ly + child.y)
        return

    if node.type == SceneNodeType.RECT or \
       node.type == SceneNodeType.BUFFER:
        width, height = node.size
        node_box = Box(lx, ly, width, height)
        if node_box.intersection(node_box, box):
            yield node


def scene_nodes_in_box(node: SceneNode, box: Box) -> Iterator[SceneNode]:
    """Recursively iterate over all child nodes intersecting `box`.

    This is basically the same as scene_nodes_in_box in wlr_scene.c, except it
    returns an iterator instead of running a callback.
    """
    _, x, y = node.coords()
    yield from _scene_nodes_in_box(node, box, x, y)


def _scale_length(length: int, offset: int, scale: float) -> int:
    return round((offset + length) * scale) - round(offset * scale)


def _scale_box(box: Box, scale: float) -> None:
    box.width = _scale_length(box.width, box.x, scale)
    box.height = _scale_length(box.height, box.y, scale)
    box.x = round(box.x * scale)
    box.y = round(box.y * scale)


def _scene_buffer_get_texture(scene_buffer: SceneBuffer, renderer: Renderer) -> Texture:
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


class SceneOutput(wlrSceneOutput):
    """Drop-in replacement for wlroots.wlr_types.SceneOutput.

    This overrides the commit() method to set up the render in Python.
    """

    def __init__(self, scene: wlrScene, output: wlrOutput, renderer: Renderer) -> None:
        super().__init__(lib.wlr_scene_output_create(scene._ptr, output._ptr))
        self.wlr_scene = scene
        self.wlr_output = output
        self.renderer = renderer

    def _scissor_output(self, box: Box) -> None:
        """Copied from scissor_output in wlr_scene.c.

        This takes a wlr_box as an input instead of a pixman_box32.
        """
        ow, oh = self.wlr_output.transformed_resolution()
        transform = wlrOutput.transform_invert(self.wlr_output.transform)
        box.transform(box, transform, ow, oh)
        self.renderer.scissor(box)

    def _render_rect(
            self, scene_rect: SceneRect, render_region: PixmanRegion32, box: Box
    ) -> None:
        """Copied from render_rect in wlr_scene.c."""
        for box in render_region.rectangles_as_boxes():
            self._scissor_output(box)
            self.renderer.render_rect(
                box, scene_rect.color, self.wlr_output.transform_matrix)

    def _render_buffer(
            self, scene_buffer: SceneBuffer, render_region: PixmanRegion32, dst_box: Box
    ) -> None:
        """Copied from render_texture and scene_node_render in wlr_scene.c.

        It accesses the private field wlr_scene_buffer.src_box.
        """
        texture = _scene_buffer_get_texture(scene_buffer, self.renderer)
        if not texture:
            return
        transform = wlrOutput.transform_invert(scene_buffer.transform)
        matrix = Matrix.project_box(
            dst_box, transform, 0.0, self.wlr_output.transform_matrix)

        src_box = ffi.addressof(scene_buffer._ptr.src_box)
        if src_box.width <= 0 or src_box.height <= 0:
            src_box = ffi.new("struct wlr_fbox *")
            src_box.x = 0
            src_box.y = 0
            src_box.width = texture.width
            src_box.height = texture.height

        for box in render_region.rectangles_as_boxes():
            self._scissor_output(box)
            self.renderer.render_subtexture_with_matrix(
                texture, src_box, matrix, 1.0)

        pwlib.wl_signal_emit(
            ffi.addressof(scene_buffer._ptr.events.output_present),
            self._ptr
        )

    def _render_node(self, node: SceneNode, damage: PixmanRegion32) -> None:
        """Copied from scene_node_render in wlr_scene.c.

        It accesses the private field wlr_scene_node.visible.
        """
        _, x, y = node.coords()
        x -= self.x
        y -= self.y

        with PixmanRegion32() as render_region:
            render_region.copy_from(node.visible)
            render_region.translate(-self.x, -self.y)
            scale = self.wlr_output.scale
            render_region.scale(render_region, scale)
            if not scale.is_integer():
                render_region.expand(render_region, 1)
            render_region.intersect(render_region, damage)
            if not render_region.not_empty():
                return

            width, height = node.size
            dst_box = Box(x, y, width, height)
            _scale_box(dst_box, scale)

            if node.type == SceneNodeType.RECT:
                self._render_rect(node.as_rect, render_region, dst_box)
            elif node.type == SceneNodeType.BUFFER:
                self._render_buffer(node.as_buffer, render_region, dst_box)

    def commit(self) -> None:
        """Based on wlr_scene_output_commit in wlr_scene.c (wlroots 0.16.2).

        This contains most of the logic in that method, including damage
        control.  Not (yet) implemented:

         * Direct scanout.
         * Background visibility tracking.
         * Damage debugging / highlighting.

        The first two are optimizations that would probably only matter on
        mobile.
        """
        output = self.wlr_output
        damage_ring = ffi.addressof(self._ptr.damage_ring)
        damage_current = PixmanRegion32(ffi.addressof(damage_ring.current))

        if not output.needs_frame and not damage_current.not_empty():
            output.rollback()
            return

        width, height = output.effective_resolution()
        output_box = Box(self.x, self.y, width, height)
        tree = self.wlr_scene.tree

        render_list = []
        for node in scene_nodes_in_box(tree.node, output_box):
            if node.invisible:
                continue
            with PixmanRegion32() as intersection:
                intersection.intersect_rect(
                    node.visible,
                    output_box.x,
                    output_box.y,
                    output_box.width,
                    output_box.height)
                if intersection.not_empty():
                    render_list.append(node)

        buffer_age = output.attach_render()

        with PixmanRegion32() as damage:
            lib.wlr_damage_ring_get_buffer_damage(
                damage_ring, buffer_age, damage._ptr)

            renderer = self.renderer
            renderer.begin(output.width, output.height)

            for box in damage.rectangles_as_boxes():
                self._scissor_output(box)
                renderer.clear([0, 0, 0, 1])

            for node in reversed(render_list):
                self._render_node(node, damage)

            self.renderer.scissor(None)
            output.render_software_cursors(damage)

            renderer.end()

        with PixmanRegion32() as frame_damage:
            tr_width, tr_height = self.wlr_output.transformed_resolution()
            transform = wlrOutput.transform_invert(output.transform)
            frame_damage.transform(damage_current, transform, tr_width, tr_height)
            output.set_damage(frame_damage)

        success = lib.wlr_output_commit(output._ptr)
        if success:
            lib.wlr_damage_ring_rotate(damage_ring)
        return success

        # super().commit()

    @property
    def x(self) -> int:
        return self._ptr.x

    @property
    def y(self) -> int:
        return self._ptr.y
