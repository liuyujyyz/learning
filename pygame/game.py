import pygame
import numpy as np
pygame.init()
clock = pygame.time.Clock()

class Club():
    def __init__(self, w = 640, h = 480):
        self._running = True
        self._display_surf = None
        self.size = self.weight, self.height = w, h
        self.x = w//2
        self.y = h//2

        self.population = 1000
        self.resource = 1000
        self.earth = 100

    def generate(self):
        self.population = int(self.population * 1.2)

    def produce(self):
        self.resource += min(self.population * 0.5, self.earth**2 * 0.1)
        self.resource = min(self.earth**2 * 10, self.resource)

    def eat(self):
        self.resource -= self.population * 0.1
        if self.resource < 0:
            self.population = int(self.population * 0.9)
            self.resource = 0
            if np.random.uniform(0,1) > 0.9:
                self.earth = int(self.earth * 0.99)

    def die(self):
        self.population = int(self.population * 0.9)

    def expand(self):
        if self.population < 10 or self.resource < 10:
            pass
        self.population -= 10
        self.resource -= 10
        self.earth += 1

    def on_init(self):
        pygame.init()
        self._display_surf = pygame.display.set_mode(self.size, pygame.HWSURFACE | pygame.DOUBLEBUF)

    def on_event(self, event):
        if event.type == pygame.QUIT:
            self._running = False

    def on_loop(self):
        self.generate()
        self.produce()
        self.eat()
        self.die()
        
        pressed = pygame.key.get_pressed()
        if pressed[pygame.K_UP]: 
            self.generate()
        if pressed[pygame.K_DOWN]: 
            self.produce()
            self.die()
        if pressed[pygame.K_LEFT]: 
            self.die()
        if pressed[pygame.K_RIGHT]:
            self.expand()

    def on_render(self):
        self._display_surf.fill((0,0,0))
        pygame.draw.circle(self._display_surf, (0, 128, min(255,int(self.population/1024))), (self.x, self.y), self.earth // 10)
        pygame.display.flip()
        pass

    def on_cleanup(self):
        pygame.quit()

    def execute(self):
        if self.on_init() == False:
            self._running = False

        while self._running:
            print(self.earth, self.population/self.earth**2, end='\r')
            for event in pygame.event.get():
                self.on_event(event)
            self.on_loop()
            self.on_render()
            clock.tick(500)
        self.on_cleanup()

if __name__ == '__main__':
    app = Club()
    app.execute()
