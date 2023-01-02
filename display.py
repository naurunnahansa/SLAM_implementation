import sdl2
import sdl2.ext

class Display(object):
    def __init__(self, W, H, WindowName):
        sdl2.ext.init()

        self.W, self.H = W,H
        self.window = sdl2.ext.Window(WindowName, size=(W,H))
        self.window.show()
        

    def draw(self, img):
        for event in sdl2.ext.get_events():
            if(event.type == sdl2.SDL_QUIT):exit(0)

        surf = sdl2.ext.pixels3d(self.window.get_surface())
        surf[:,:,0:3] = img.swapaxes(0,1)
        self.window.refresh()