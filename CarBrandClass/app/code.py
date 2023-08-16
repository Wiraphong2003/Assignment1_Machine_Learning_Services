import re
import pickle

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

# h[[]]
# m = pickle.load(open(r'model\imageCAR_model.pkl','rb'))
# print(predict_Car(m,h))
