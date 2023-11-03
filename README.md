# microsoft / kosmos-2 Cog model

This is an implementation of the [microsoft/kosmos-2](https://huggingface.co/microsoft/kosmos-2-patch14-224) as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

First, download the pre-trained weights:

    cog run script/download-weights

Then, you can run predictions:

    cog predict -i image=@snowman.png

## Example:

```
An image of a snowman warming himself by a campfire.

[('a snowman', (12, 21), [(0.390625, 0.046875, 0.984375, 0.828125)]), ('a campfire', (41, 51), [(0.109375, 0.640625, 0.546875, 0.984375)])]
```

![alt text](snowman.png)