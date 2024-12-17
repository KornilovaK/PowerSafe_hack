# PowerSafe_hack

## Команда misis arbyz:
* [Екатерина Корнилова](https://github.com/KornilovaK) - MLE, CV, дизайнер, капитан
* [Крылов Александр](https://github.com/realalexkrylov) - MLE, CV
* Дмитрий Бусыгин - MLE, CV
* Аксинья Харьюзова - дизайнер

## Проблема
Древесно-кустарниковая растительность (ДКР) в охранных зонах воздушных линий электропередачи приводит к аварийным ситуациям и перебоям в электроснабжении.

## Задача
Разработать алгоритм компьютерного зрения, способный обнаруживать ДКР в охранных зонах ЛЭП на основе облаков точек. Чтобы выполнить задачу хакатона, нужно также распознавать опоры линий электропередачи по типу — промежуточные и анкерные, токоведущие провода с целью недопущения негативного влияния ДКР на работу ЛЭП.


## Наше решение
1. Сжатие данных. Было: 128 GB, Стало: 2.75 GB без существенной потери информативности. С помощью библиотек laspy и open3d voxel_down_sample уменьшили размерность облаков точек на 40 процентов. Файл - **make_data.py**. Готовые датасеты на Hugging Face (в файлах также находится csv с информацией о том, с какого индекса в датасете начинается определенный .las файл):
- [Train](https://huggingface.co/datasets/Eka-Korn/power_line_lidar_data)
- [Test](https://huggingface.co/datasets/Eka-Korn/power_line_lidar_data_test)

**Пример из Train датасета:** 
> ![Пример датасета](https://github.com/user-attachments/assets/9d3ec8ca-d9d0-48bb-ba79-f53243fa9e26)

**Пример .las файла местности:**
> ![Пример .las файла](https://github.com/user-attachments/assets/5a7438ca-d373-4b11-884f-2e6e4a0a196e)

2. Подход - object detection. Делаем custom dataset, где каждому сжатого файла сопостаставляем все объекты, относящиеся к нему. batch_size = 1, так как каждый сэмпл очень большой
```
Сlass CustomDataSet(DataSet):
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
        return len(self.counts_df)
```

3. Обучение. Как первую (и последнюю, потому что не успели) модель использовали простейшую архитектуру Conv Net 1D, предсказывая alpha*b bounding boxes, где b - количество объектов на каждом .las файле. Файл - **first_model.pth**
> ![image](https://github.com/user-attachments/assets/630dba42-20fc-4077-abc0-ff4a03568b9d)

Каждый бокс представляет из себя массив из 10 значений: 7 для регресии (cx, cy, cz, dx, dy, dz, yaw) и 3 для классификации (так как всего 3 класса). В качестве лосс функции соединили SmoothL1Loss (для регрессии; лучше для object detection, чем MSE) и CrossEntropyLoss (для многоклассовой классификации)

```
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.criterion_reg = nn.SmoothL1Loss()
        self.criterion_class = nn.CrossEntropyLoss()
        
    def forward(self, boxes, preds):
        right_preds = []
        # as batch_size = 1
        boxes = boxes[0]
        for box in boxes:
            target_class = torch.tensor(box[-1], dtype=torch.long)
            target_xyz = torch.tensor(box[:-1], dtype=torch.float)
        
            min_loss, l = 10**9, -1
            for i, pred in enumerate(preds):
                pred_classes = pred[-3:]
                pred_coordinates = pred[:-3]
                loss_reg = self.criterion_reg(pred_coordinates, target_xyz)
                loss_class = self.criterion_class(pred_classes, target_class)
                loss = loss_reg# + loss_class
                if loss < min_loss:
                    min_loss = loss
                    l = i
                    
            right_preds.append(preds[l].unsqueeze(0))
            preds = torch.concatenate([preds[:l, :], preds[(l+1):, :]], dim=0)
        
        right_preds = torch.concatenate(right_preds, dim=0)
        
        target_class = torch.tensor(boxes[:, -1], dtype=torch.long)
        target_xyz = torch.tensor(boxes[:, :-1], dtype=torch.float)
        pred_classes = right_preds[:, -3:]
        pred_coordinates = right_preds[:, :-3]
        
        loss_reg = self.criterion_reg(pred_coordinates, target_xyz)
        loss_class = self.criterion_class(pred_classes, target_class)
        return loss_reg, loss_class
```

4. Инференс. Предсказываем n самых вероятных "боксов". Для вычисления числа n натренировали модельку для нахождения зависимости n от размера .las файла, но в итогом решении нашли этот параметром подбором

5. Что планировалось для использования, но не успелось :(
* PyTorch Geometric
* GGN
* 3DETR
* CenterNet
* PointNet
* и тд
