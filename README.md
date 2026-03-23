# Пути нужно будет поправить на  свои

Для инференса созаем виртуальное окружение с phyton 3.11 и устанавливаем requiments_iner.txt

Для тренировки модели создаем еще одно виртуальное окружение с phyton 3.7 и устанавливаем requiments_train.txt
            # Находясь в  виртуальном окружении с phyton 3.7, изменить количество  эпох и запустить:
python banknote_net/src/train_custom.py --data_path ./data_money --enc_path ./banknote_net/models/banknote_net_encoder.h5 --bsize 8 --epochs 30

Для разметки зон в кадре запускаем zone_editor.py

Для раскадровки видео используем extract_frames_enhanced.py

Для сохранения  рук используем empty_h_.py

Структура папок для тренировки  banknote_net
project/
├── data_money/
│   ├── train/               # фото для  тренировки
│   ├── val/         # эталонные фото
