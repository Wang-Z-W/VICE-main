# WaterBird-Edit

WaterBird-Edit probes whether multimodal knowledge edits follow bird habitat evidence across different backgrounds. `edit_annotations.json` contains **1,136 editing instances: 636 train and 500 test**.

## Annotations

Each instance contains an edit image and target answer, a text rephrase, an image rephrase of the same species, and text/image locality queries.

| Fields | Description |
| --- | --- |
| `image`, `src`, `alt` | Edit image, question, and target habitat (`water` or `land`) |
| `water`, `place`, `species`, `split` | Habitat label, edit-image background, species, and train/test split |
| `rephrase` | Rephrased edit question |
| `image_rephrase`, `image_rephrase_place` | Same-species probe image and its background |
| `loc_q`, `loc_a` | Text locality question and answer |
| `m_loc_img`, `m_loc_q`, `m_loc_a` | Image locality query |

The diagnostic groups use **(target habitat, image-rephrase background)**:

| Split | WW | WL | LW | LL |
| --- | --- | --- | --- | --- |
| Train | 123 | 146 | 197 | 170 |
| Test | 115 | 93 | 133 | 159 |

## Images

Download the source images:

- **CUB-200-2011:** [Images and annotations](https://data.caltech.edu/records/65de6-vp158).
- **MMEdit:** [Image archive](https://drive.google.com/file/d/1fQzJBFkok5kFZT6QUuT-HCuYKk2Vb93O/view), provided in the [official dataset instructions](https://github.com/zjunlp/EasyEdit/blob/main/examples/MMEdit.md).

Image paths are relative to the repository root. Arrange the downloaded images as follows:

```text
data/
  CUB_200_2011/CUB_200_2011/images/<species>/<filename>.jpg
  easy-edit-mm/images/val2014/<filename>.jpg
```
