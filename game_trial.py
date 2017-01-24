import os
import math
import pygame
from pygame.locals import *

SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
MAX_TILTER_ANGLE = 30
VERT_MAX_SPEED = 10

def load_png(name, scale=None):
    """ Load image and return image object"""
    fullname = os.path.join('images', name)
    try:
        image = pygame.image.load(fullname)
        if image.get_alpha() is None:
            image = image.convert()
        else:
            image = image.convert_alpha()
        if scale:
            image = pygame.transform.scale(image, (scale))
    except pygame.error, message:
            print 'Cannot load image:', fullname
            raise SystemExit, message
    return image, image.get_rect()

class Tilter(pygame.sprite.Sprite):
    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self)
        self.img_width = 200
        self.img_height = 50
        self.image, self.rect = load_png('tilter.png', (self.img_width, self.img_height))
        self.x = x - self.img_width/2
        self.y = y - self.img_height/2
        self.elevation = SCREEN_HEIGHT - self.y
        self.angle = 0
        self.max_input_angle = math.degrees(math.atan(float(SCREEN_HEIGHT/2 - 0) / SCREEN_WIDTH * 0.8))
        self.max_output_angle = MAX_TILTER_ANGLE

    def update(self, mouse_position):
        self.angle = math.degrees(math.atan(float(SCREEN_HEIGHT/2 - mouse_position[1]) / SCREEN_WIDTH * 0.8))
        self.angle *= self.max_output_angle / self.max_input_angle
        self.elevation += self.angle * VERT_MAX_SPEED / self.max_output_angle
        self.elevation = min(self.elevation, SCREEN_HEIGHT-self.img_height)
        self.elevation = max(self.elevation, self.img_height)
        self.y = SCREEN_HEIGHT - self.elevation

    def draw(self, surface):
        image = pygame.transform.rotate(self.image, self.angle)
        self.y -= abs((math.tan(math.radians(self.angle)) * self.img_width) / 2)# half change in height
        surface.blit(image, (self.x, self.y))


def main():
    # Initialise screen
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption('Tilty')

    # Fill background
    background = pygame.Surface(screen.get_size())
    background = background.convert()
    background.fill((100, 100, 100))

    # Display the tilter
    tilter = Tilter(SCREEN_WIDTH*.2, SCREEN_HEIGHT/2)
    # trackersprite = pygame.sprite.RenderPlain(tracker)

    # Blit everything to the screen
    screen.blit(background, (0, 0))
    pygame.display.flip()

    clock = pygame.time.Clock()

    # Event loop
    while 1:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == QUIT:
                return

        pos = pygame.mouse.get_pos()
        tilter.update(pos)

        screen.blit(background, (0,0))
        tilter.draw(screen)
        pygame.display.flip()


if __name__ == '__main__':
    main()
