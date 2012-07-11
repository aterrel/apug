import numpy as np
import Image
import pygame
from pygame.locals import *
import sys
import cevent
from mandel_cl import calc_fractal_opencl

calc_fractal = calc_fractal_opencl

class MandelApp(cevent.CEvent):
    def __init__(self, h=512, w=512):
        self.window_size = (h, w)
        self._running = True
        self._display_surf = None
        self._image_surf = None
        self.view = np.array([-2.13, 0.77, -1.3, 1.3])
        self.down = None
        self.zoom_rate = 50
        self.move_rate = 50
        self.maxiter = 30
        self.changed = True

    def draw_mandelbrot(self, x1, x2, y1, y2, maxiter=30):
        # draw the Mandelbrot set, from numpy example
        h,w = self.window_size
        xx = np.arange(x1, x2, (x2-x1)/w)
        yy = np.arange(y2, y1, (y1-y2)/h) * 1j
        q = np.ravel(xx+yy[:, np.newaxis]).astype(np.complex64)
        output = calc_fractal(q, maxiter)
        max_out = float(output.max()) * 255.
        return (output.reshape((h,w)) /
                float(output.max()) * 255.).astype(np.uint8)

    def on_init(self):
        pygame.init()
        msg = "KEYS: Arrows: to move,  i/o: zoom in and out,  u/d: to up and down iter count,  p: to save to file"
        font_obj = pygame.font.SysFont(pygame.font.get_default_font(), 24)
        self.msg_surf = font_obj.render(msg, True, (255,255,255))
        self.msg_rect = self.msg_surf.get_rect()
        self.msg_rect.topleft = (0,0)
        self._display_surf = pygame.display.set_mode(self.window_size, pygame.HWSURFACE)
        self._running = True

    def on_loop(self):
        if self.down is None:
            return
        xdiff =  (self.view[1] - self.view[0])
        ydiff =  (self.view[3] - self.view[2])

        if self.down == K_i:
            xdiff /= self.zoom_rate
            ydiff /= self.zoom_rate
            self.view[0::2] += (xdiff, ydiff)
            self.view[1::2] -= (xdiff, ydiff)
        elif self.down == K_o:
            xdiff /= self.zoom_rate
            ydiff /= self.zoom_rate
            self.view[0::2] -= (xdiff, ydiff)
            self.view[1::2] += (xdiff, ydiff)
        if self.down == K_UP:
            self.view[2:] += ydiff / self.move_rate
        if self.down == K_RIGHT:
            self.view[:2] += xdiff / self.move_rate
        if self.down == K_DOWN:
            self.view[2:] -= ydiff / self.move_rate
        if self.down == K_LEFT:
            self.view[:2] -= xdiff / self.move_rate

    def on_render(self):
        if not self.down and not self.changed:
            return

        mandel = self.draw_mandelbrot(*self.view, maxiter=self.maxiter)
        self.im = Image.fromarray(mandel)
        self.im.putpalette(reduce(
                lambda a,b: a+b, ((i,0,0) for i in range(255))
            ))
        im = self.im.convert("RGB")
        self._image_surf = pygame.image.fromstring(im.tostring(), im.size, im.mode).convert()

        self._display_surf.blit(self._image_surf,(0,0))
        self._display_surf.blit(self.msg_surf, (0, 0))
        pygame.display.flip()

    def on_exit(self):
        self._running = False

    def on_key_up(self, event):
        self.down = None

    def on_key_down(self, event):
        if event.key in [K_i, K_o, K_UP, K_RIGHT, K_DOWN, K_LEFT]:
            self.down = event.key

        elif event.key == K_p :
            print("Printing to file")
            self.im.save("mandel.png")

        elif event.key == K_u:
            self.maxiter += 10

        elif event.key == K_d:
            self.maxiter -= 10

        else:
            return
        self.changed = True

    def on_cleanup(self):
        pygame.quit()

    def on_execute(self):
        if self.on_init() == False:
            self._running = False

        while( self._running ):
            for event in pygame.event.get():
                self.on_event(event)
            self.on_loop()
            self.on_render()
        self.on_cleanup()

if __name__ == "__main__" :

    MandelApp(w = 512*2, h = 512*2).on_execute()
