import random
import uuid

standalone_names = [
    "Strawberry_Serenity",
    "Lemon_Drop",
    "Figsaw",
    "Grape_Digger",
    "Blood_Orange",
    "Fuego",
    "Planet_of_the_Grapes",
    "Jinjer",
    "Burstberry",
    "Fruit_Punch"
    "Legendberry",
    "Dingleberries",
    "Pear_Pressure",
    "Berry_Blast",
    "Pearanoid",
    "Meloncholy",
    "Grapeful",
    "Existentialist_Apricot",
    "Oetinger_Pils",
    "Pilsator_Platin",
    "Sternburger_Export",
    "Karate_Angriff",
    "Raspberry_Pi",
    "Banana_Pi",
    "Mango_Curry",
]

fruits = [
    "Strawberry",
    "Lemon",
    "Lime",
    "Peach",
    "Orange",
    "Apple",
    "Banana",
    "Apricot",
    "Coconut",
    "Lychee",
    "Acai",
    "Papaya",
    "Passionfruit",
    "Jackfruit",
    "Pomelo",
    "Kiwi",
    "Cranberry",
    "Blackberry",
    "Cherry"
]

colors = [
    "Black",
    "Silver",
    "Gray",
    "Red",
    "Green",
    "Yellow",
    "Coral",
    "Pink",
    "Gold",
    "Rosybrown",
    "Skyblue",
    "Vantablack",
    "Purple"
]


def generate_name() -> str:
    """
    just a fuction to generate not boring names, that can be used as ids.
    Its only benefit is that it looks nicer that plain uuids.

    :return: a fancy string
    """
    num = random.randint(0, 100)
    if random.uniform(0, 1) > 0.8:
        return f"{random.choice(standalone_names)}_{num:02d}_{str(uuid.uuid4())[:8]}"
    else:
        return f"{random.choice(colors)}_{random.choice(fruits)}_{num:02d}_{str(uuid.uuid4())[:8]}"


if __name__ == '__main__':
    for i in range(20):
        print(generate_name())
