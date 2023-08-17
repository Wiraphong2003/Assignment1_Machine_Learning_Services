import pickle
import json
brand = {
    1:"Audi",
    2:"Hyundai Creta",
    3:"Mahindra Scorpio",
    4:"Rolls Royce",
    5:"Swift",
    6:"Tata Safari",
    7:"Toyota Innova"
}

def predictcar(m,HOG):
    result = m.predict(HOG)
    return brand[result[0]]


# with open("app\hogtest.json", "r") as json_file:
#     data = json.load(json_file)


# hots = data['Hog']
# m = pickle.load(open(r'model/image_modelv2.pk', 'rb'))
# print(predictcar(m,[hots]))

