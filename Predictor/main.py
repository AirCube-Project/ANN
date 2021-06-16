import numpy as np
import matplotlib.pyplot as plt

data = {"gender": "F", "age": 22, "maritalStatus": "в отношениях", "educationLevel": "Бакалавриат", "awakeTime": "8:0",
        "sleepTime": "0:0", "events": [
        {"name": "Завтрак", "step": 0, "start": 510, "duration": 30, "id": "fb0c4e60-7f57-43e3-95c1-1ae4ba77cf39",
         "scores": [5, 5, 5, 3, 1, 1, 4, 1]},
        {"name": "Обед", "step": 1, "start": 870, "duration": 30, "id": "f811da8f-0e50-436e-9a44-bb3e1dcf5c06",
         "scores": [5, 4, 5, 3, 2, 1, 4, 1]},
        {"name": "Ужин", "step": 2, "start": 1155, "duration": 45, "id": "816c9566-7562-4c97-8218-8e25614b7a8b",
         "scores": [4, 4, 4, 4, 3, 1, 4, 1]},
        {"name": "Принять душ", "step": 4, "start": 1350, "duration": 45, "id": "3d9ad591-60f7-4898-92b3-0a9a35e72948",
         "scores": [5, 5, 5, 5, 1, 1, 4, 1]}, {"name": "Почистить зубы", "step": 5, "start": 495, "duration": 15,
                                               "id": "351f585f-3122-4b1d-a36a-8b3df719defc",
                                               "scores": [5, 5, 5, 2, 1, 1, 4, 1]},
        {"name": "Мытье посуды", "step": 7, "start": 840, "duration": 15, "id": "b775777d-2923-4722-8b0c-a4e41b1b6158",
         "scores": [5, 4, 5, 1, 1, 1, 3, 1]},
        {"name": "Мытье посуды", "step": 7, "start": 1080, "duration": 15, "id": "eb0cd4ea-843d-447b-bafb-bffec020c67e",
         "scores": [5, 4, 5, 1, 1, 1, 3, 1]}, {"name": "Приготовление пищи", "step": 8, "start": 855, "duration": 15,
                                               "id": "658ceeb0-07e8-4428-bf27-e719447299f9",
                                               "scores": [5, 5, 5, 4, 1, 1, 4, 3]},
        {"name": "Приготовление пищи", "step": 8, "start": 1095, "duration": 60,
         "id": "45a81518-7fd3-418b-aed6-96751c4eef9d", "scores": [5, 5, 5, 4, 1, 1, 4, 3]},
        {"name": "Приготовление пищи", "step": 8, "start": 510, "duration": 10,
         "id": "aaf96a00-a505-4424-b4d0-fb8839d84371", "scores": [5, 5, 5, 4, 1, 1, 4, 3]},
        {"name": "Просмотр почты", "step": 10, "start": 975, "duration": 15,
         "id": "78643c90-ec7d-48ab-bcb4-d8430805ae03", "scores": [4, 5, 4, 2, 1, 3, 1, 2]},
        {"name": "Просмотр почты", "step": 10, "start": 645, "duration": 15,
         "id": "a23e851a-f876-44c9-a83e-b903717049e2", "scores": [4, 5, 4, 2, 1, 3, 1, 2]},
        {"name": "Просмотр новостей", "step": 12, "start": 645, "duration": 15,
         "id": "f4e7ddaf-f5cb-4429-ad09-3e1b8a217797", "scores": [4, 5, 5, 3, 1, 2, 2, 3]},
        {"name": "Просмотр новостей", "step": 12, "start": 975, "duration": 15,
         "id": "7769400c-44ba-4e1e-9db2-a8aa2cfabcea", "scores": [4, 5, 5, 3, 1, 2, 2, 3]},
        {"name": "Чтение художественной литературы", "step": 15, "start": 525, "duration": 30,
         "id": "db494be9-35de-4aa7-a024-c2589a0ed36c", "scores": [3, 4, 4, 5, 1, 2, 3, 2]},
        {"name": "Чтение художественной литературы", "step": 15, "start": 870, "duration": 30,
         "id": "11a4b063-c156-48b9-aa8f-13f8d0d38bac", "scores": [3, 4, 4, 5, 1, 2, 3, 2]},
        {"name": "Чтение художественной литературы", "step": 15, "start": 1155, "duration": 45,
         "id": "6e4ca368-89bf-4041-a5ec-cd1677e9c787", "scores": [3, 4, 4, 5, 1, 2, 3, 2]},
        {"name": "Чтение художественной литературы", "step": 15, "start": 1425, "duration": 15,
         "id": "1047a560-21d3-48bb-b226-c6cd9e5b72cd", "scores": [3, 4, 4, 5, 1, 2, 3, 2]},
        {"name": "Составление презентации", "step": 18, "start": 675, "duration": 120,
         "id": "79bc98a5-1228-44b2-b38b-b0c338713f3a", "scores": [2, 3, 4, 3, 5, 4, 3, 3]},
        {"name": "Составление презентации", "step": 18, "start": 930, "duration": 135,
         "id": "a37e0b8b-38dd-4321-9313-a6b659dbe7b8", "scores": [2, 3, 4, 3, 5, 4, 3, 3]},
        {"name": "Составление презентации", "step": 18, "start": 1215, "duration": 90,
         "id": "29c46355-3377-489d-84e7-7e54b23568e8", "scores": [2, 3, 4, 3, 5, 4, 3, 3]},
        {"name": "Прогулка", "step": 24, "start": 540, "duration": 15, "id": "6c17a49d-0275-42e7-87c0-d7e11c100c95",
         "scores": [5, 4, 5, 5, 1, 1, 4, 1]},
        {"name": "Прогулка", "step": 24, "start": 570, "duration": 15, "id": "8db7385d-fd96-4c21-9b9f-f9726360c726",
         "scores": [5, 4, 5, 5, 1, 1, 4, 1]},
        {"name": "Прогулка", "step": 24, "start": 810, "duration": 15, "id": "9c7f1c53-32eb-448f-8fb1-4ac920c897e2",
         "scores": [5, 4, 5, 5, 1, 1, 4, 1]},
        {"name": "Прогулка", "step": 24, "start": 1065, "duration": 15, "id": "752db055-b591-40ec-964d-f2cc5cbcbfb1",
         "scores": [5, 4, 5, 5, 1, 1, 4, 1]},
        {"name": "Прогулка", "step": 24, "start": 1200, "duration": 15, "id": "bb85d664-763d-4fbb-b4ce-5fd62d22b725",
         "scores": [5, 4, 5, 5, 1, 1, 4, 1]},
        {"name": "Прогулка", "step": 24, "start": 1320, "duration": 30, "id": "5bd4364c-37de-4e14-a2ff-ea212278f335",
         "scores": [5, 4, 5, 5, 1, 1, 4, 1]}, {"name": "Путь на работу", "step": 26, "start": 555, "duration": 30,
                                               "id": "1115e50e-a2f8-42fa-9cea-694f74c24a51",
                                               "scores": [5, 4, 5, 2, 1, 1, 2, 1]},
        {"name": "Путь на работу", "step": 26, "start": 540, "duration": 15,
         "id": "255c9361-400f-4b01-9bad-4a3e0f287d4b", "scores": [5, 4, 5, 2, 1, 1, 2, 1]},
        {"name": "Путь домой", "step": 27, "start": 780, "duration": 45, "id": "03bf3032-aeab-4d80-8dbd-a5fee592cfdc",
         "scores": [5, 5, 5, 2, 1, 1, 2, 1]},
        {"name": "Путь домой", "step": 27, "start": 1020, "duration": 65, "id": "d681d5fc-41eb-41c6-83a8-6a37442d9c3f",
         "scores": [5, 5, 5, 2, 1, 1, 2, 1]},
        {"name": "Совещание", "step": 29, "start": 1260, "duration": 45, "id": "29f0b048-214f-4d65-800c-78673cd9ee99",
         "scores": [3, 5, 4, 2, 2, 4, 2, 3]},
        {"name": "Взаимодействие с соцсетями", "step": 30, "start": 480, "duration": 15,
         "id": "1beeacb4-38ee-49a2-9194-028c59ecef75", "scores": [5, 5, 5, 3, 1, 2, 2, 2]},
        {"name": "Взаимодействие с соцсетями", "step": 30, "start": 630, "duration": 15,
         "id": "f9a70c2c-689c-413e-a1d5-4be51170d9f7", "scores": [5, 5, 5, 3, 1, 2, 2, 2]},
        {"name": "Взаимодействие с соцсетями", "step": 30, "start": 825, "duration": 15,
         "id": "3afd3d84-ce7d-4580-a282-2cf913ef19d7", "scores": [5, 5, 5, 3, 1, 2, 2, 2]},
        {"name": "Взаимодействие с соцсетями", "step": 30, "start": 900, "duration": 15,
         "id": "3941a8b8-b138-4851-af53-1cabb20c5acf", "scores": [5, 5, 5, 3, 1, 2, 2, 2]},
        {"name": "Взаимодействие с соцсетями", "step": 30, "start": 960, "duration": 15,
         "id": "19dcbdb7-12f9-4858-8718-fc082b9166c3", "scores": [5, 5, 5, 3, 1, 2, 2, 2]},
        {"name": "Взаимодействие с соцсетями", "step": 30, "start": 1030, "duration": 30,
         "id": "763486ed-a409-4cea-9e2f-cfd9576772c6", "scores": [5, 5, 5, 3, 1, 2, 2, 2]},
        {"name": "Взаимодействие с соцсетями", "step": 30, "start": 790, "duration": 25,
         "id": "09cc1225-e5fe-409f-a5b4-4e5397f7773e", "scores": [5, 5, 5, 3, 1, 2, 2, 2]},
        {"name": "Взаимодействие с соцсетями", "step": 30, "start": 555, "duration": 20,
         "id": "879b6dcd-65a2-4025-9c14-6524c6d48981", "scores": [5, 5, 5, 3, 1, 2, 2, 2]},
        {"name": "Взаимодействие с соцсетями", "step": 30, "start": 1230, "duration": 15,
         "id": "437803d4-23e0-43d1-b49e-7c32a2d65fd2", "scores": [5, 5, 5, 3, 1, 2, 2, 2]},
        {"name": "Взаимодействие с соцсетями", "step": 30, "start": 1410, "duration": 15,
         "id": "4b3be4a4-9b29-480a-891d-3193002a9702", "scores": [5, 5, 5, 3, 1, 2, 2, 2]},
        {"name": "Прослушивание музыки", "step": 33, "start": 540, "duration": 45,
         "id": "ce8b091a-2248-4228-a8e7-eeef08cd9163", "scores": [5, 5, 5, 5, 1, 2, 3, 4]},
        {"name": "Прослушивание музыки", "step": 33, "start": 615, "duration": 225,
         "id": "fef6c411-ac5c-462e-8364-81c80667a0d5", "scores": [5, 5, 5, 5, 1, 2, 3, 4]},
        {"name": "Прослушивание музыки", "step": 33, "start": 915, "duration": 210,
         "id": "5d528a0e-2233-4ef2-a52b-caefb5cf154d", "scores": [5, 5, 5, 5, 1, 2, 3, 4]},
        {"name": "Прослушивание музыки", "step": 33, "start": 1215, "duration": 90,
         "id": "f1dcf804-6800-42ee-8e09-2ba586be7016", "scores": [5, 5, 5, 5, 1, 2, 3, 4]},
        {"name": "Прослушивание музыки", "step": 33, "start": 1365, "duration": 60,
         "id": "a0617927-84b5-471f-8002-49c84a299f63", "scores": [5, 5, 5, 5, 1, 2, 3, 4]},
        {"name": "Написание кода программы", "step": 35, "start": 585, "duration": 150,
         "id": "5119a728-10d7-4ab0-9742-97e4852894d3", "scores": [2, 4, 4, 4, 4, 5, 3, 5]},
        {"name": "Составление методики обучения для детей 5 класса", "step": 39, "start": 585, "duration": 255,
         "id": "6646d060-a78f-484d-ab13-cf5a5549fefa", "scores": [2, 4, 3, 4, 5, 5, 3, 4]},
        {"name": "Составление методики обучения для детей 5 класса", "step": 39, "start": 915, "duration": 150,
         "id": "4032c2d6-685b-45de-8096-e03ada50281d", "scores": [2, 4, 3, 4, 5, 5, 3, 4]},
        {"name": "Составление методики обучения для детей 5 класса", "step": 39, "start": 1215, "duration": 90,
         "id": "b6c17256-e993-49be-a979-ae29b55f550a", "scores": [2, 4, 3, 4, 5, 5, 3, 4]}]}

interval_scores = []
scoring_scale = []

shift = 15
for tod in range(0, 24 * 60 // shift):
    interval_scores.append([])
    scoring_scale.append(0)
awake = data["awakeTime"]
aw_split = awake.split(":")
awake_time = (int)(aw_split[0]) * 60 + (int)(aw_split[1])
sleep = data["sleepTime"]
aw_split = sleep.split(":")
sleep_time = (int)(aw_split[0]) * 60 + (int)(aw_split[1])
if awake_time < sleep_time:
    awake_time = 24 * 60 + awake_time
for tod in range(sleep_time // shift, awake_time // shift):
    print(tod)
    interval_scores[tod % (24 * 60 // shift)] = [-1, -1, -1, -1, -1, -1, -1, -1]
    scoring_scale[tod % (24 * 60 // shift)] = 1
print(interval_scores)

if sleep_time < awake_time:
    sleep_time = 24 * 60 + sleep_time

for tod in range(awake_time // shift, sleep_time // shift):
    interval_scores[tod % ((24 * 60) // shift)] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    scoring_scale[tod % ((24 * 60) // shift)] = 0
    for event in data["events"]:
        if event['start'] > tod * shift:  # начинается позже
            continue
        if event['start'] + event['duration'] <= tod * shift:  # заканчивается раньше
            continue
        if event['start'] <= tod * shift and event['start'] + event['duration'] >= tod * shift + shift:
            # целиком во временном слоте
            for i in range(0, 8):
                interval_scores[tod % (24 * 60 // shift)][i] += event["scores"][i]
            scoring_scale[tod % (24 * 60 // shift)] += 1
        else:
            print(event['start'])
            print(tod * shift)
            print(event['start'] + event['duration'])
            print(tod * shift + shift)
            start = event["start"]
            if start < tod * shift:
                start = tod * shift
            finish = event["start"] + event["duration"]
            if finish > tod * shift + shift:
                finish = tod * shift + shift
            delta = (finish - start)
            for i in range(0, 8):
                interval_scores[tod % (24 * 60 // shift)][i] += event["scores"][i] * (delta / shift)
            scoring_scale[tod % (24 * 60 // shift)] += delta / shift

print(scoring_scale)
for i in range(0, len(scoring_scale)):
    for j in range(0, 8):
        if scoring_scale[i % ((24 * 60) // shift)] != 0:
            interval_scores[i % (24 * 60 // shift)][j] /= scoring_scale[i % (24 * 60 // shift)]

zeros = []
for j in range(0, 8):
    zeros.append([])
for i in range(0, len(scoring_scale)):
    for j in range(0, 8):
        if i != 0 and interval_scores[i - 1][j] == interval_scores[i][j] and interval_scores[i][j] != -1:
            zeros[j].append(i)
for z in range(0,8):
    for i in zeros[z]:
        interval_scores[i][z] = 0
# interpolate!
for j in range(0, 1):
    data = []
    for i in range(0, len(scoring_scale)):
        if interval_scores[i][j] > 0:
            data.append([i * shift, interval_scores[i][j]])
    x = np.array(list(map(lambda d: d[0], data)))
    y = np.array(list(map(lambda d: d[1], data)))
    print(x)


