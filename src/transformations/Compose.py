class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, size):
        for t in self.transforms:
            if type(t).__name__ == "Resize":
                img = t(img, size)
            else:
                img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string