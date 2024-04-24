# Дообучение предобученной модели для классификации изображений
Использовался датасет [Animals-10](https://www.kaggle.com/datasets/alessiocorrado99/animals10) с Kaggle.

Были дообучены модели [ShuffleNetV2_0_5, ShuffleNetV2_1_0](https://pytorch.org/vision/stable/models/shufflenetv2.html) и [EfficientNetB0](https://pytorch.org/vision/stable/models/efficientnet.html)

Сравнение моделей  
<img src=models_comparison.jpg width=900>  
Лучше всего модель EfficientNetB0, однако она и самая большая - 4млн параметров, ShuffleNetV2_1_0 - 1.2млн параметров, ShuffleNetV2_0_5 - 0.3млн параметров.

# Результаты предсказаний ShuffleNetV2_0_5
<table>
   <tr>
      <td><img src=predictions/chims.jpg width=800></td>
   </tr>
   <tr>
      <td><img src=predictions/chiro.jpg width=800></td>
   </tr>
   <tr>
      <td><img src=predictions/crunchycat.jpg width=800></td>
   </tr>
   <tr>
      <td><img src=predictions/maxwell.jpg width=800></td>
   </tr>
   <tr>
      <td><img src=predictions/nelson.jpg width=800></td>
   </tr>
</table>
