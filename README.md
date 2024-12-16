# PowerSafe_hack

## Команда misis arbyz

### Екатерина Корнилова - MLE и CV- специалист, дизайнер
### Крылов Александр - MLE и CV- специалист
### Дмитрий Бусыгин - MLE и CV- специалист
### Аксинья Харьюзова - дизайнер

* Первым делом с помощью  open3d и laspy мы сделали downsamling 3d вокселей и сохранили на hugging face один цельный датасет, разделили на train/val выборки и сохранили значения индексов, с каких начинаются какие .las файлы. Датасет[https://huggingface.co/datasets/Eka-Korn/power_line_lidar_data]
* Точно так же обработавли тестовые (публичный датасет) [https://huggingface.co/datasets/Eka-Korn/power_line_lidar_data_test]

* В качестве архитектуры использовали custom Conv Net 1D, предсказывая alpha*b, где b - количество объектов на каждом .las файле. Модель предсказывает 10 значений: 7 для регресии, 3 для классификации (тк 3 класса)
* В качестве лосса соединили SmoothL1Loss и CrossEntropyLoss
* В ходе инференса предсказываем n самых вероятных "боксов". Для вычисления числа n натренировали модельку для нахождения зависимости n от размера .las файла






## Проблема
Древесно-кустарниковая растительность (ДКР) в охранных зонах воздушных линий электропередачи приводит к аварийным ситуациям и перебоям в электроснабжении.

## Задача
Разработать алгоритм компьютерного зрения, способный обнаруживать ДКР в охранных зонах ЛЭП на основе облаков точек. Чтобы выполнить задачу хакатона, нужно также распознавать опоры линий электропередачи по типу — промежуточные и анкерные, токоведущие провода с целью недопущения негативного влияния ДКР на работу ЛЭП.


## Как мы решили данную задачу?

### 1. Сжали данные
С помощью open3d voxel_down_sample уменьшили размерность облаков точек на 40 процентов, что позволило сохранить все файлы в одном датасете и выложить на Hugging face ![image](https://github.com/user-attachments/assets/9d3ec8ca-d9d0-48bb-ba79-f53243fa9e26)


Было: 128 GB, 
Стало: 2.75 GB![image](https://github.com/user-attachments/assets/5a7438ca-d373-4b11-884f-2e6e4a0a196e)

### 2. Сделали датасеты для сэмплирования на каждый .las файл всех объектов,  относящихся к нему. batch_size = 1, так как каждый сэмпл очень большой
```Сlass CustomDataSet(DataSet):
    def __init__(self, sub_ds, counts_df, new_df):
        self.ds = sub_ds
        self.counts_df = counts_dfself.new_df = new_df
    def __getitem__(self,i):
        row = self.counts_df.iloc[i]
        start, end = row[0], row[1]
 cur_df = self.ds[start:end]
 cur_df = pd.DataFrame(cur_df).sort_values(by=’x’).reset_index(drop=True).T
 cur_df = torch.tensor(cur_df.values, dtype = tourch.float)
 boxes = self.new_df[self.new_df[‘file name’] == row[‘index’]][‘answer’].values[0]
 return cur_df, boxes
 def __len__(self):
 return len(self.counts_df)```



Наши попытки:
 PointNe
 PyTorch Geometri
 GN
 3DET
 CenterNe
 Paddle3
 Point Transformer
 Learning3
 и так далее.

Пытались написать свое подобие object detection, но сжатые сроки не позволили



### Архитектура решения.
![image](https://github.com/user-attachments/assets/630dba42-20fc-4077-abc0-ff4a03568b9d)
