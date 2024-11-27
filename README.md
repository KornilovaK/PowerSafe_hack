# PowerSafe_hack

* Первым делом с помощью  open3d и laspy мы сделали downsamling 3d вокселей и сохранили на hugging face один цельный датасет, разделили на train/val выборки и сохранили значения индексов, с каких начинаются какие .las файлы. Датасет[https://huggingface.co/datasets/Eka-Korn/power_line_lidar_data]
* Точно так же обработавли тестовые (публичный датасет) [https://huggingface.co/datasets/Eka-Korn/power_line_lidar_data_test]

* В качестве архитектуры использовали custom Conv Net 1D, предсказывая alpha*b, где b - количество объектов на каждом .las файле. Модель предсказывает 10 значений: 7 для регресии, 3 для классификации (тк 3 класса)
* В качестве лосса соединили SmoothL1Loss и CrossEntropyLoss
