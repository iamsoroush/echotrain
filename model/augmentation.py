import albumentations as A


class Augmentation:

    """
    This class is implementing augmentation on the batches of data

    Example::

        aug = Augmentation()
        data = aug.batch_augmentation(x,y)
        x = images(batch)
        y = masks(batch)
        data = augmented batch

    """

    def __init__(self, config=None):

        """
        :param config: augmentation part of config file: config.pre_process.augmentation, containing:
          rotation_range - the range limitation for rotation in augmentation
          flip_proba - probability for flipping
        """

        self._load_params(config)

        self.transform = A.Compose([
            A.Flip(p=self.flip_proba),
            A.ShiftScaleRotate(0, 0, border_mode=0, rotate_limit=self.rotation_range, p=self.rotation_proba)
        ])

    def batch_augmentation(self, batch):

        """This method implement augmentation on batches

        :param batch: (x, y):
            x: batch images of the whole batch
            y: batch masks of the whole batch

        :return x: image batch
        :return y: mask batch.
        """

        # changing the type of the images for albumentation
        x = batch[0]
        y = batch[1]

        x = x.astype('float32')

        # implementing augmentation on every image and mask of the batch
        for i in range(len(x)):
            transformed = self.transform(image=x[i], mask=y[i])
            x[i] = transformed['image']
            y[i] = transformed['mask']

        return x, y

    def add_augmentation(self, generator):

        """Calling the batch_augmentation

        :param generator: the input of this class must be generator

        :yield: the batches of the augmented generator
        """

        while True:
            batch = next(generator)
            augmented_batch = self.batch_augmentation(batch)
            yield augmented_batch

    def _load_params(self, config):
        self._set_defaults()

        if config is not None:
            aug_config = config.pre_process.augmentation

            self.rotation_range = aug_config.rotation_range
            self.rotation_proba = aug_config.rotation_proba
            self.flip_proba = aug_config.flip_proba

    def _set_defaults(self):
        self.rotation_range = 45
        self.rotation_proba = 0.5
        self.flip_proba = 0.5
