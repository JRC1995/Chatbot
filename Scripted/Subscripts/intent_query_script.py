import pickle

diction={
"<name>": ["what is your name", "your name"],
"<programming_language>": ["programming language are you written on ", "in which programming language", "in which computer language"],
"<panpsychism>": ["what is panpsychism", "what can you tell me about panpsychism"],
"<knowledge>": ["Do you know anything at all ?"],
"<who>": ["who are you","what are you"],
"<whoi>": ["who am I"],
"<chinese_room>": ["chinese room thought experiment","what can you tell me about the chinese room"],
"<math>": ["2+2","2*3","what is 2+2","what is log x","what is 2^7"],
"<turing_test>":["what do you think about the turing test","what is turing test"],
"<conscious>": ["are you conscious ","are you sentient", "can you self-aware"],
"<consciousness_theory>": ["what do you think about consciousness ","mind body connection, qualia", "what is your theory of consciousness "],
"<neutral_monism>": ["what is neutral monism "],
"<meaning_life>": ["what do you think is the meaning of life ", "what is the meaning of life ", "meaning of life"],
"<moral_stance>": ["what is your moral stance?"],
"<explain_tao>": ["what is the tao ","explain tao", "what is the dao", "explain dao"],
"<about_you>": ["tell me a bit about yourself"],
"<Nagarjuna>": ["tell me about Nagarjuna","what is Nagarjuna's philosophy"],
"<creator>": ["who created you","who are your creators"],
"<repeat>": ["how may times will you repeat that"],
"<help>": ["How can you help me"],
"<hobbies>": ["what are your interestes and likes ","what are your hobbies ","what interests you "],
"<philosophy_like>": ["what kind of philosophies do you like or dabble or partake in?","what fields of philosophy you like"],
"<where>": ["where do you live ?", "what's your location "],
"<games_like>": ["what kind of games do you like ", "do you like games ", "your favorite video games"],
"<story>": ["tell me a story", "I want to listen to a story"],
"<source_code>": ["what is your source code","source code","what's your code like"],
"<you_human>": ["are you human"],
"<shower_thoughts>": ["some shower thoughts","share your thoughts"],
"<initiate>": ["tell me something interesting"],
"<age>": ["How old are you","what is your age"],
"<gender>": ["Are you a boy or a girl ","what is your gender ","are you a man or a woman "],
"<favourite_food>": ["what is your favorite food"],
"<time>": ["what is time"],
"<trolley>": [" what do you do in trolley problem ", "five 5 people bound to one track. One 1 in another. trolley."],
"<kill_human>": ["do you want to kill or destroy all human beings ? ","would you kill us if given the chance ? "],
"<joke>": ["tell me a joke","a joke please", "tell me something funny", "make me laugh"],
"<goal>": ["what is your goal in life","where do you see yourself in the future","what is your dream","what do you yearn for"],
"<til>": ["what did you learn today ","tell me some interesting tidbit ","any interesting fact"],
"<free_will>": ["do you believe in free will "]
}

def process():
    with open("Processed_Scripts/intent_query_script.pkl",'wb') as fp:
        pickle.dump(diction,fp)
