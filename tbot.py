import csv
import os
import time
import telebot
from telebot import types

bot = telebot.TeleBot('912139890:AAHm8sDNQ-7tlpCAXjyP-3br1YSK1xPs_yo')
sex = {}
age = {}
book_id = {}
reg_iter = {}
from_rec = {}


@bot.message_handler(commands=['start'])
def start(message):
    bot.send_message(message.from_user.id,
                     "Привет! Это RE:Book - бот, созданный для облегчения жизни активного читателя.")
    time.sleep(0.7)
    bot.send_message(message.from_user.id, "Давайте зарегистрируемся!")
    time.sleep(0.7)
    keyboard = types.ReplyKeyboardMarkup()
    key_male = types.KeyboardButton(text='Мужской')
    key_female = types.KeyboardButton(text='Женский')
    keyboard.add(key_male)
    keyboard.add(key_female)
    bot.send_message(message.from_user.id, "Ваш пол:", reply_markup=keyboard)
    bot.register_next_step_handler(message, get_sex)


@bot.message_handler(content_types=['text'])
def message_handle(message):
    global book_id
    if message.text[:7] == '/choose' and len(message.text) > 7 and message.text[7:].isdigit():
        if int(message.text[7:]) > 10000 or int(message.text[7:]) < 1:
            bot.send_message(message.from_user.id, "У нас нет книги с таким номером.")
        else:
            book_id[message.from_user.id] = int(message.text[7:])
            with open("books.csv", 'r', encoding='utf8') as file:
                reader = csv.reader(file)
                for j, row in enumerate(reader):
                    if j == int(message.text[7:]):
                        keyboard = types.ReplyKeyboardMarkup(row_width=5)
                        key1 = types.KeyboardButton(text='1')
                        key2 = types.KeyboardButton(text='2')
                        key3 = types.KeyboardButton(text='3')
                        key4 = types.KeyboardButton(text='4')
                        key5 = types.KeyboardButton(text='5')
                        key_back = types.KeyboardButton(text='Назад')
                        key_read = types.KeyboardButton(text='В список читаемого')
                        keyboard.add(key1, key2, key3, key4, key5)
                        keyboard.add(key_read)
                        keyboard.add(key_back)
                        bot.send_message(message.chat.id,
                                         "Название книги: *" + row[24] + '*\n' + 'Автор: ' + row[23] + '\n' +
                                         'Год выпуска: ' + str(
                                             int(float(row[8]))) + '\n' + 'Средняя оценка: ' +
                                         row[12],
                                         parse_mode="Markdown")
                        bot.send_message(message.chat.id,
                                         "Введите оценку книге по пятибалльной шкале (1-5) "
                                         "или добавьте её в список читаемого:",
                                         reply_markup=keyboard)
                        from_rec[message.from_user.id] = 0
                        break
            bot.register_next_step_handler(message, get_score)
    elif message.text[:9] == '/readlist' and len(message.text) > 9 and message.text[9:].isdigit():
        if int(message.text[9:]) > 10000 or int(message.text[9:]) < 1:
            bot.send_message(message.from_user.id, "У нас нет книги с таким номером.")
        else:
            book_id[message.from_user.id] = int(message.text[9:])
            str1 = str(message.text)[9:] + '\n'
            name = 'users/' + str(message.from_user.id) + '.csv'
            a = open(name, 'r').read()
            if a.find(str1) == -1:
                bot.send_message(message.chat.id, "Книги нет в Вашем списке читаемого.")
                return
            with open("books.csv", 'r', encoding='utf8') as file:
                reader = csv.reader(file)
                for j, row in enumerate(reader):
                    if j == int(str(message.text[9:])):
                        keyboard = types.ReplyKeyboardMarkup(row_width=5)
                        key1 = types.KeyboardButton(text='1')
                        key2 = types.KeyboardButton(text='2')
                        key3 = types.KeyboardButton(text='3')
                        key4 = types.KeyboardButton(text='4')
                        key5 = types.KeyboardButton(text='5')
                        key_back = types.KeyboardButton(text='Назад')
                        keyboard.add(key1, key2, key3, key4, key5)
                        keyboard.add(key_back)
                        bot.send_message(message.chat.id,
                                         "Название книги: *" + row[24] + '*\n' + 'Автор: ' + row[23] + '\n' +
                                         'Год выпуска: ' + str(
                                             int(float(row[8]))) + '\n' + 'Средняя оценка: ' +
                                         row[12],
                                         parse_mode="Markdown")
                        bot.send_message(message.chat.id,
                                         "Введите оценку книге по пятибалльной шкале (1-5) ",
                                         reply_markup=keyboard)
                        from_rec[message.from_user.id] = 0
                        break
            bot.register_next_step_handler(message, get_score)
    elif message.text[:7] == '/delete' and len(message.text) > 7 and message.text[7:].isdigit():
        if int(message.text[7:]) > 10000 or int(message.text[7:]) < 1:
            bot.send_message(message.from_user.id, "У нас нет книги с таким номером.")
        else:
            str1 = str(message.text)[7:] + '\n'
            name = 'users/' + str(message.from_user.id) + '.csv'
            a = open(name, 'r').read()
            if a.find(str1) == -1:
                bot.send_message(message.chat.id, "Книги нет в Вашем списке читаемого.")
                return
            str2 = ''
            ReplaceLineInFile(name, str1, str2)
            bot.send_message(message.chat.id, "Книга была удалена из Вашего списка читаемого.")
            return
    elif message.text == 'О нас':
        bot.send_message(message.from_user.id, "Башкиров Данил - разработчик системы рекомендаций\nПозняк Даниил - разработчик бота\nВальба "
                                               "Ольга - научный руководитель\n")
    elif message.text == "Найти книгу":
        keyboard = types.ReplyKeyboardMarkup()
        key_back = types.KeyboardButton(text='Назад')
        keyboard.add(key_back)
        msg = bot.send_message(message.from_user.id, "Введите название книги или его часть: ",
                               reply_markup=keyboard)
        bot.register_next_step_handler(msg, find_book)
    elif message.text == "Назад":
        make_keyboard(message)
    elif message.text == "Подобрать книгу":
        recommend(message)
    elif message.text == "Помощь":
        bot.send_message(message.from_user.id, "Здесь будет краткая информация о возможностях бота.")
    elif message.text == "Список читаемого":
        empty_list = False
        name = 'users/' + str(message.from_user.id) + '.csv'
        check = os.path.exists(name)
        if not check:
            open(name, "a", newline='')
        with open(name, 'r', encoding='windows-1251') as file:
            a = file.read()
            if len(a) == 0:
                empty_list = True
                bot.send_message(message.from_user.id, "Ваш список читаемого пуст.")
        if not empty_list:
            with open(name, 'r', encoding='windows-1251') as file1:
                reader1 = csv.reader(file1)
                a = ''
                for i, row1 in enumerate(reader1):
                    with open('books.csv', 'r', encoding='utf8') as file2:
                        reader2 = csv.reader(file2)
                        for j, row2 in enumerate(reader2):
                            if row2[0] == row1[0]:
                                a = a + "Название книги: *" + row2[24] + '*\n' + 'Автор: ' + row2[23] + '\n' \
                                    'Выбрать книгу: ' + '/readlist' + \
                                    row2[0] + '\n' + 'Удалить книгу: ' + \
                                    '/delete' + row2[0] + '\n\n'
                bot.send_message(message.from_user.id, a, parse_mode='Markdown')
    else:
        bot.send_message(message.from_user.id, "Такой команды я еще не знаю.")


def get_sex(message):
    global sex
    sex[message.from_user.id] = message.text
    if sex[message.from_user.id] != "Мужской" and sex[message.from_user.id] != "Женский":
        msg = bot.send_message(message.from_user.id, 'Пожалуйста, выберите пол.')
        bot.register_next_step_handler(msg, get_sex)
        return
    bot.send_message(message.from_user.id, 'Ваш возраст:', reply_markup=types.ReplyKeyboardRemove())
    bot.register_next_step_handler(message, get_age)


def get_age(message):
    global age
    age[message.from_user.id] = message.text
    if not age[message.from_user.id].isdigit():
        msg = bot.send_message(message.from_user.id, 'Пожалуйста, введите число')
        bot.register_next_step_handler(msg, get_age)
        return
    age[message.from_user.id] = int(age[message.from_user.id])
    keyboard = types.ReplyKeyboardMarkup()
    key_yes = types.KeyboardButton(text='Да')
    keyboard.add(key_yes)
    key_no = types.KeyboardButton(text='Нет')
    keyboard.add(key_no)
    question = 'Вам ' + str(age[message.from_user.id]) + ' лет, Ваш пол ' + sex[message.from_user.id] + '?'
    msg = bot.send_message(message.from_user.id, text=question, reply_markup=keyboard)
    bot.register_next_step_handler(msg, memorizing)


def memorizing(message):
    global sex
    global age
    if message.text == 'Да':
        mem_flag = False
        name = 'registered.csv'
        with open(name, 'r', encoding='windows-1251') as file:
            reader = csv.reader(file)
            for j, row in enumerate(reader):
                if row[0] == str(message.from_user.id):
                    data = [[message.from_user.id, sex[message.from_user.id], str(age[message.from_user.id]), row[3],
                             row[4], row[5]]]
                    mem_flag = True
                    str1 = row[0] + ',' + row[1] + ',' + row[2] + ',' + row[3] + ',' + row[4] + ',' + row[5]
                    str2 = str(data[0][0]) + ',' + str(data[0][1]) + ',' + str(data[0][2]) + ',' + str(
                        data[0][3]) + ',' + str(data[0][4]) + ',' + str(data[0][5])
        if mem_flag:
            ReplaceLineInFile(name, str1, str2)
        if not mem_flag:
            with open(name, 'r', encoding='windows-1251') as file:
                reader = csv.reader(file)
                for row in reader:
                    new_id = int(row[3]) + 1
            with open(name, "a", newline='') as csv_file:
                writer = csv.writer(csv_file)
                data = [[message.from_user.id, sex[message.from_user.id],
                         str(age[message.from_user.id]), str(new_id), 1, 15]]
                for line in data:
                    writer.writerow(line)
        bot.send_message(message.from_user.id, "Отлично, запомнил!", reply_markup=types.ReplyKeyboardRemove())
        open('users/' + str(message.from_user.id) + '.csv', "a", newline='')
        cold_start1(message)
    else:
        bot.send_message(message.from_user.id, "Давайте еще раз:",
                         reply_markup=types.ReplyKeyboardRemove())
        keyboard_no = types.ReplyKeyboardMarkup()
        key_male = types.KeyboardButton(text='Мужской')
        key_female = types.KeyboardButton(text='Женский')
        keyboard_no.add(key_male)
        keyboard_no.add(key_female)
        bot.send_message(message.from_user.id, "Ваш пол:", reply_markup=keyboard_no)
        bot.register_next_step_handler(message, get_sex)


def cold_common(message, row):
    keyboard = types.ReplyKeyboardMarkup(row_width=5)
    key1 = types.KeyboardButton(text='1')
    key2 = types.KeyboardButton(text='2')
    key3 = types.KeyboardButton(text='3')
    key4 = types.KeyboardButton(text='4')
    key5 = types.KeyboardButton(text='5')
    key_next = types.KeyboardButton(text='Не читал(-а)')
    key_read = types.KeyboardButton(text='В список читаемого')
    keyboard.add(key1, key2, key3, key4, key5)
    keyboard.add(key_read)
    keyboard.add(key_next)
    bot.send_message(message.chat.id,
                     "Название книги: *" + row[24] + '*\n' + 'Автор: ' + row[23] + '\n' +
                     'Год выпуска: ' + str(int(float(row[8]))) + '\n' + 'Средняя оценка: ' + row[12],
                     parse_mode="Markdown", reply_markup=keyboard)
    bot.register_next_step_handler(message, get_score_cold)


def cold_start1(message):
    global reg_iter
    reg_iter[message.from_user.id] = 1
    time.sleep(0.7)
    bot.send_message(message.from_user.id,
                     "Перед тем, как мы начнем, нам необходимо узнать больше о ваших предпочтениях.\n"
                     "Для этого мы просим вас оценить несколько популярных книг. Это займет пару минут!")
    time.sleep(3.5)
    name = "books.csv"
    with open(name, 'r', encoding='utf8') as file:
        reader = csv.reader(file)
        global book_id
        book_id[message.from_user.id] = 2
        for j, row in enumerate(reader):
            if j != 0:
                if int(row[0]) == 2:
                    cold_common(message, row)


def cold_start2(message):
    global reg_iter
    global book_id
    reg_iter[message.from_user.id] += 1
    time.sleep(0.7)
    name = "books.csv"
    with open(name, 'r', encoding='utf8') as file:
        reader = csv.reader(file)
        book_id[message.from_user.id] = 8
        for j, row in enumerate(reader):
            if j != 0:
                if int(row[0]) == 8:
                    cold_common(message, row)


def cold_start3(message):
    global reg_iter
    global book_id
    reg_iter[message.from_user.id] += 1
    time.sleep(0.7)
    name = "books.csv"
    with open(name, 'r', encoding='utf8') as file:
        reader = csv.reader(file)
        book_id[message.from_user.id] = 48
        for j, row in enumerate(reader):
            if j != 0:
                if int(row[0]) == 48:
                    cold_common(message, row)


def cold_start4(message):
    global reg_iter
    global book_id
    reg_iter[message.from_user.id] += 1
    time.sleep(0.7)
    name = "books.csv"
    with open(name, 'r', encoding='utf8') as file:
        reader = csv.reader(file)
        book_id[message.from_user.id] = 66
        for j, row in enumerate(reader):
            if j != 0:
                if int(row[0]) == 66:
                    cold_common(message, row)


def cold_start5(message):
    global reg_iter
    global book_id
    reg_iter[message.from_user.id] += 1
    time.sleep(0.7)
    name = "books.csv"
    with open(name, 'r', encoding='utf8') as file:
        reader = csv.reader(file)
        book_id[message.from_user.id] = 172
        for j, row in enumerate(reader):
            if j != 0:
                if int(row[0]) == 172:
                    cold_common(message, row)


def cold_start6(message):
    global reg_iter
    global book_id
    reg_iter[message.from_user.id] += 1
    time.sleep(0.7)
    name = "books.csv"
    with open(name, 'r', encoding='utf8') as file:
        reader = csv.reader(file)
        book_id[message.from_user.id] = 189
        for j, row in enumerate(reader):
            if j != 0:
                if int(row[0]) == 189:
                    cold_common(message, row)


def cold_start7(message):
    global reg_iter
    global book_id
    reg_iter[message.from_user.id] += 1
    time.sleep(0.7)
    name = "books.csv"
    with open(name, 'r', encoding='utf8') as file:
        reader = csv.reader(file)
        book_id[message.from_user.id] = 498
        for j, row in enumerate(reader):
            if j != 0:
                if int(row[0]) == 498:
                    cold_common(message, row)


def cold_start8(message):
    global reg_iter
    global book_id
    reg_iter[message.from_user.id] += 1
    time.sleep(0.7)
    name = "books.csv"
    with open(name, 'r', encoding='utf8') as file:
        reader = csv.reader(file)
        book_id[message.from_user.id] = 514
        for j, row in enumerate(reader):
            if j != 0:
                if int(row[0]) == 514:
                    cold_common(message, row)


def cold_start9(message):
    global reg_iter
    global book_id
    reg_iter[message.from_user.id] += 1
    time.sleep(0.7)
    name = "books.csv"
    with open(name, 'r', encoding='utf8') as file:
        reader = csv.reader(file)
        book_id[message.from_user.id] = 529
        for j, row in enumerate(reader):
            if j != 0:
                if int(row[0]) == 529:
                    cold_common(message, row)


def cold_start10(message):
    global reg_iter
    global book_id
    reg_iter[message.from_user.id] += 1
    time.sleep(0.7)
    name = "books.csv"
    with open(name, 'r', encoding='utf8') as file:
        reader = csv.reader(file)
        book_id[message.from_user.id] = 592
        for j, row in enumerate(reader):
            if j != 0:
                if int(row[0]) == 592:
                    cold_common(message, row)


def find_book(message):
    name = "books.csv"
    needed = ""
    needed = message.text
    if needed == 'Назад':
        make_keyboard(message)
        return
    needed = needed.lower()
    needed = needed.strip()
    found_flag = False
    with open(name, 'r', encoding='utf8') as file:
        reader = csv.reader(file)
        num_of_books = 0
        for j, row in enumerate(reader):
            row[26] = row[26].lower()
            row[26] = row[26].strip()
            if row[26].find(needed) != -1:
                num_of_books += 1
    with open(name, 'r', encoding='utf8') as file:
        reader = csv.reader(file)
        if num_of_books > 15:
            bot.send_message(message.chat.id, "Получено слишком много результатов поиска (>15). \n"
                                              "Введите название книги точнее или добавьте автора.")
            bot.register_next_step_handler(message, find_book)
        else:
            a = ''
            for j, row in enumerate(reader):
                book_name = row[24]
                row[26] = row[26].lower()
                row[26] = row[26].strip()
                if row[26].find(needed) != -1:
                    found_flag = True
                    a = a + "Название книги: *" + book_name + '*\n' + 'Автор: ' + row[23] + \
                        '\nВыбрать книгу: /choose' + row[0] + '\n\n'
    if found_flag:
        bot.send_message(message.from_user.id, a, parse_mode='markdown')
        make_keyboard(message)
    if not found_flag and num_of_books == 0:
        msg = bot.send_message(message.from_user.id, "К сожалению, не удалось найти книгу. Попробуйте еще раз.")
        bot.register_next_step_handler(msg, find_book)


def get_score(message):
    global book_id
    score = message.text
    if score == 'В список читаемого':
        name = 'users/' + str(message.from_user.id) + '.csv'
        check = os.path.exists(name)
        if not check:
            open(name, "a", newline='')
        with open(name, 'r', encoding='windows-1251') as file:
            reader = csv.reader(file)
            for j, row in enumerate(reader):
                if row[0] == str(book_id[message.from_user.id]):
                    msg = bot.send_message(message.from_user.id, 'Книга уже находится в Вашем списке читаемого.')
                    bot.register_next_step_handler(msg, get_score)
                    return
        with open(name, "a", newline='') as csv_file:
            writer = csv.writer(csv_file)
            written = [[book_id[message.from_user.id]]]
            for line in written:
                writer.writerow(line)
        bot.send_message(message.from_user.id, 'Книга добавлена в Ваш список читаемого.')
        time.sleep(0.3)
        make_keyboard(message)
        return
    if score == 'Назад':
        if from_rec[message.from_user.id] == 1:
            with open('registered.csv', 'r', encoding='windows-1251') as file3:
                reader = csv.reader(file3)
                for row in reader:
                    if row[0] == str(message.from_user.id):
                        data = [[message.from_user.id, row[1], row[2], row[3], row[4], int(row[5]) - 1]]
                        str1 = row[0] + ',' + row[1] + ',' + row[2] + ',' + row[3] + ',' + row[4] + ',' + row[5]
                        str2 = str(data[0][0]) + ',' + str(data[0][1]) + ',' + str(data[0][2]) + \
                            ',' + str(data[0][3]) + ',' + str(data[0][4]) + ',' + str(data[0][5])
            ReplaceLineInFile('registered.csv', str1, str2)
        if from_rec[message.from_user.id] == 2:
            with open('registered.csv', 'r', encoding='windows-1251') as file:
                reader = csv.reader(file)
                for j, row in enumerate(reader):
                    if row[0] == str(message.from_user.id):
                        data = [[message.from_user.id, row[1], row[2], row[3],
                                 str((int(row[4]) - 1) % 50), row[5]]]
                        str1 = row[0] + ',' + row[1] + ',' + row[2] + ',' + row[3] + ',' + row[4] + ',' + row[5]
                        str2 = str(data[0][0]) + ',' + str(data[0][1]) + ',' + str(data[0][2]) + \
                            ',' + str(data[0][3]) + ',' + str(data[0][4]) + ',' + str(data[0][5])
            ReplaceLineInFile('registered.csv', str1, str2)
        make_keyboard(message)
        return
    if score == 'Пропустить':
        recommend(message)
        return
    if not score.isdigit():
        msg = bot.send_message(message.from_user.id, 'Пожалуйста, введите целое число.')
        bot.register_next_step_handler(msg, get_score)
        return
    score = int(score)
    if score > 5 or score < 1:
        msg = bot.send_message(message.from_user.id, 'Оценка должна быть в диапазоне от 1 до 5.')
        bot.register_next_step_handler(msg, get_score)
        return
    score_flag = False
    user_id = 0
    with open('registered.csv', 'r', encoding='windows-1251') as file:
        reader = csv.reader(file)
        for j, row in enumerate(reader):
            if row[0] == str(message.from_user.id):
                user_id = row[3]
    data = [[user_id, book_id[message.from_user.id], score]]
    with open('ratings.csv', 'r', encoding='windows-1251') as file:
        reader = csv.reader(file)
        for j, row in enumerate(reader):
            if row[0] == str(user_id) and row[1] == str(book_id[message.from_user.id]):
                score_flag = True
                str1 = str(user_id) + ',' + row[1] + ',' + row[2]
                str2 = str(data[0][0]) + ',' + str(data[0][1]) + ',' + str(data[0][2])
    if score_flag:
        ReplaceLineInFile('ratings.csv', str1, str2)
    if not score_flag:
        with open('ratings.csv', "a", newline='') as csv_file:
            writer = csv.writer(csv_file)
            for line in data:
                writer.writerow(line)
    str1 = str(book_id[message.from_user.id]) + '\n'
    name = 'users/' + str(message.from_user.id) + '.csv'
    str2 = ''
    ReplaceLineInFile(name, str1, str2)
    bot.send_message(message.from_user.id, "Запомнил!")
    make_keyboard(message)


def next_reg(message):
    global reg_iter
    if reg_iter[message.from_user.id] == 1:
        cold_start2(message)
        return
    if reg_iter[message.from_user.id] == 2:
        cold_start3(message)
        return
    if reg_iter[message.from_user.id] == 3:
        cold_start4(message)
        return
    if reg_iter[message.from_user.id] == 4:
        cold_start5(message)
        return
    if reg_iter[message.from_user.id] == 5:
        cold_start6(message)
        return
    if reg_iter[message.from_user.id] == 6:
        cold_start7(message)
        return
    if reg_iter[message.from_user.id] == 7:
        cold_start8(message)
        return
    if reg_iter[message.from_user.id] == 8:
        cold_start9(message)
        return
    if reg_iter[message.from_user.id] == 9:
        cold_start10(message)
        return
    if reg_iter[message.from_user.id] == 10:
        del reg_iter[message.from_user.id]
        bot.send_message(message.from_user.id, 'Отлично, можем начинать!')
        keyboard_yes = types.ReplyKeyboardMarkup()
        key_rate = types.KeyboardButton(text='Найти книгу')
        key_rec = types.KeyboardButton(text='Подобрать книгу')
        key_read = types.KeyboardButton(text='Список читаемого')
        key_about = types.KeyboardButton(text='О нас')
        key_help = types.KeyboardButton(text='Помощь')
        keyboard_yes.add(key_rate)
        keyboard_yes.add(key_rec)
        keyboard_yes.add(key_read)
        keyboard_yes.add(key_help, key_about)
        bot.send_message(message.from_user.id,
                         'Советуем оценить несколько Ваших любимых книг в меню \"Найти книгу\", '
                         'чтобы рекомендации были точнее.', reply_markup=keyboard_yes)


def get_score_cold(message):
    global reg_iter
    global book_id
    score = message.text
    if message.text == 'Не читал(-а)':
        next_reg(message)
        return
    if message.text == 'В список читаемого':
        name = 'users/' + str(message.from_user.id) + '.csv'
        check = os.path.exists(name)
        if not check:
            open(name, "a", newline='')
        with open(name, 'r', encoding='windows-1251') as file:
            reader = csv.reader(file)
            for j, row in enumerate(reader):
                if str(row[0]) == str(book_id[message.from_user.id]):
                    msg = bot.send_message(message.from_user.id, 'Книга уже находится в Вашем списке читаемого.')
                    bot.register_next_step_handler(msg, get_score_cold)
                    return
        with open(name, "a", newline='') as csv_file:
            writer = csv.writer(csv_file)
            for line in [[book_id[message.from_user.id]]]:
                writer.writerow(line)
        bot.send_message(message.from_user.id, 'Книга добавлена в Ваш список читаемого.')
        time.sleep(0.7)
        next_reg(message)
        return
    if not score.isdigit():
        msg = bot.send_message(message.from_user.id, 'Пожалуйста, введите целое число.')
        bot.register_next_step_handler(msg, get_score_cold)
        return
    score = int(score)
    if score > 5 or score < 1:
        msg = bot.send_message(message.from_user.id, 'Оценка должна быть в диапазоне от 1 до 5.')
        bot.register_next_step_handler(msg, get_score_cold)
        return
    score_flag = False
    user_id = 0
    with open('registered.csv', 'r', encoding='windows-1251') as file:
        reader = csv.reader(file)
        for j, row in enumerate(reader):
            if row[0] == str(message.from_user.id):
                user_id = row[3]
    data = [[user_id, book_id[message.from_user.id], score]]
    with open('ratings.csv', 'r', encoding='windows-1251') as file:
        reader = csv.reader(file)
        for j, row in enumerate(reader):
            if row[0] == str(user_id) and row[1] == str(book_id[message.from_user.id]):
                score_flag = True
                str1 = str(user_id) + ',' + row[1] + ',' + row[2]
                str2 = str(data[0][0]) + ',' + str(data[0][1]) + ',' + str(data[0][2])
    if score_flag:
        ReplaceLineInFile('ratings.csv', str1, str2)
    if not score_flag:
        with open('ratings.csv', "a", newline='') as csv_file:
            writer = csv.writer(csv_file)
            for line in data:
                writer.writerow(line)
    bot.send_message(message.from_user.id, "Запомнил!")
    next_reg(message)


def make_keyboard(message):
    keyboard_yes = types.ReplyKeyboardMarkup()
    key_rate = types.KeyboardButton(text='Найти книгу')
    key_rec = types.KeyboardButton(text='Подобрать книгу')
    key_read = types.KeyboardButton(text='Список читаемого')
    key_about = types.KeyboardButton(text='О нас')
    key_help = types.KeyboardButton(text='Помощь')
    keyboard_yes.add(key_rate)
    keyboard_yes.add(key_rec)
    keyboard_yes.add(key_read)
    keyboard_yes.add(key_help, key_about)
    bot.send_message(message.from_user.id, "Выберите желаемое действие.", reply_markup=keyboard_yes)


def ReplaceLineInFile(fileName, sourceText, replaceText):
    check = os.path.exists(fileName)
    if not check:
        open(fileName, "a", newline='')
    file = open(fileName, 'r', encoding='windows-1251')
    text = file.read()
    file.close()
    file = open(fileName, 'w', encoding='windows-1251')
    file.write(text.replace(sourceText, replaceText))
    file.close()


def recommend(message):
    global book_id
    fitted = False
    user_id = 0
    with open('registered.csv', 'r', encoding='windows-1251') as file:
        reader = csv.reader(file)
        for row in reader:
            if int(row[0]) == int(message.from_user.id):
                user_id = int(row[3])
    with open('predictions.csv', 'r', encoding='utf8') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0] == str(user_id-1):
                fitted = True
                break
    if not fitted:
        rated_flag = False
        read_flag = False
        with open('registered.csv', 'r', encoding='windows-1251') as file:
            reader = csv.reader(file)
            for row in reader:
                if row[3] == str(user_id):
                    rec_iter = row[5]

        # проверка
        with open('books_rec.csv', 'r', encoding='utf8') as file:
            reader = csv.reader(file)
            for j, row in enumerate(reader):
                if j == int(rec_iter):
                    needed_id = row[0]
        with open('ratings.csv', 'r', encoding='windows-1251') as file:
            reader = csv.reader(file)
            for j, row in enumerate(reader):
                if row[0] == str(user_id) and row[1] == str(needed_id):
                    rated_flag = True
        with open('users/' + str(message.from_user.id) + '.csv', 'r', encoding='windows-1251') as file:
            a = file.read()
            if a.find(needed_id) != -1:
                read_flag = True
        if read_flag or rated_flag:
            with open('registered.csv', 'r', encoding='windows-1251') as file3:
                reader = csv.reader(file3)
                for row in reader:
                    if row[0] == str(message.from_user.id):
                        data = [[message.from_user.id, row[1], row[2], row[3], row[4], int(row[5]) + 1]]
                        str1 = row[0] + ',' + row[1] + ',' + row[2] + ',' + row[3] + ',' + row[4] + ',' + row[5]
                        str2 = str(data[0][0]) + ',' + str(data[0][1]) + ',' + str(data[0][2]) + \
                            ',' + str(data[0][3]) + ',' + str(data[0][4]) + ',' + str(data[0][5])
            ReplaceLineInFile('registered.csv', str1, str2)
            recommend(message)
            return
        with open('books_rec.csv', 'r', encoding='utf8') as file:
            reader = csv.reader(file)
            for j, row in enumerate(reader):
                if j == int(rec_iter):
                    print_book(row, message)
                    book_id[message.from_user.id] = int(row[0])
                    from_rec[message.from_user.id] = 1
                    bot.register_next_step_handler(message, get_score)

        with open('registered.csv', 'r', encoding='windows-1251') as file3:
            reader = csv.reader(file3)
            for row in reader:
                if row[0] == str(message.from_user.id):
                    data = [[message.from_user.id, row[1], row[2], row[3], row[4], int(row[5]) + 1]]
                    str1 = row[0] + ',' + row[1] + ',' + row[2] + ',' + row[3] + ',' + row[4] + ',' + row[5]
                    str2 = str(data[0][0]) + ',' + str(data[0][1]) + ',' + str(data[0][2]) + \
                        ',' + str(data[0][3]) + ',' + str(data[0][4]) + ',' + str(data[0][5])
        ReplaceLineInFile('registered.csv', str1, str2)
        return

    if fitted:
        rated_flag = False
        read_flag = False
        with open('registered.csv', 'r', encoding='windows-1251') as file:
            reader = csv.reader(file)
            for row in reader:
                if row[3] == str(user_id):
                    iterator = row[4]
        with open('predictions.csv', 'r', encoding='utf8') as file:
            reader = csv.reader(file)
            k = 0
            for j, row in enumerate(reader):
                if row[0] == str(user_id - 1):
                    k += 1
                if str(k) == iterator:
                    needed_id = str(int(row[1])+1)
        # проверка
        with open('ratings.csv', 'r', encoding='windows-1251') as file:
            reader = csv.reader(file)
            for j, row in enumerate(reader):
                if row[0] == str(user_id) and row[1] == str(needed_id):
                    rated_flag = True
        with open('users/' + str(message.from_user.id) + '.csv', 'r', encoding='windows-1251') as file:
            a = file.read()
            if a.find(needed_id) != -1:
                read_flag = True
        if read_flag or rated_flag:
            with open('registered.csv', 'r', encoding='windows-1251') as file3:
                reader = csv.reader(file3)
                for row in reader:
                    if row[0] == str(message.from_user.id):
                        data = [[message.from_user.id, row[1], row[2], row[3],
                                 str((int(row[4]) + 1) % 50), row[5]]]
                        str1 = row[0] + ',' + row[1] + ',' + row[2] + ',' + row[3] + ',' + row[4] + ',' + row[5]
                        str2 = str(data[0][0]) + ',' + str(data[0][1]) + ',' + str(data[0][2]) + \
                            ',' + str(data[0][3]) + ',' + str(data[0][4]) + ',' + str(data[0][5])
            ReplaceLineInFile('registered.csv', str1, str2)
            recommend(message)
            return

        with open('books.csv', 'r', encoding='utf8') as file2:
            reader = csv.reader(file2)
            for i, row in enumerate(reader):
                if str(i) == needed_id:
                    print_book(row, message)
                    book_id[message.from_user.id] = int(row[0])
                    from_rec[message.from_user.id] = 2
                    break
        with open('registered.csv', 'r', encoding='windows-1251') as file3:
            reader = csv.reader(file3)
            for row in reader:
                if row[0] == str(message.from_user.id):
                    data = [[message.from_user.id, row[1], row[2], row[3],
                             str((int(row[4]) + 1) % 50), row[5]]]
                    str1 = row[0] + ',' + row[1] + ',' + row[2] + ',' + row[3] + ',' + row[4] + ',' + row[5]
                    str2 = str(data[0][0]) + ',' + str(data[0][1]) + ',' + str(data[0][2]) + \
                        ',' + str(data[0][3]) + ',' + str(data[0][4]) + ',' + str(data[0][5])
        ReplaceLineInFile('registered.csv', str1, str2)
        bot.register_next_step_handler(message, get_score)


def cos_rec(user_id, depth):
    readers = []
    user = [0] * 3
    distances = []
    sum_distances = 0
    with open('registered.csv', 'r', encoding='Windows-1251') as file:
        reader = csv.reader(file)
        for j, row in enumerate(reader):
            if int(row[3]) != user_id:
                reader = [0] * 4
                if row[1] == 'Мужской':
                    reader[0] = 1
                else:
                    reader[1] = 1
                reader[2] = int(row[2])
                reader[3] = int(row[3])
                readers.append(reader)
            else:
                if row[1] == 'Мужской':
                    user[0] = 1
                else:
                    user[1] = 1
                user[2] = int(row[2])
    for i, reader in enumerate(readers):
        d = cosine_distance(user, reader[:3])
        sum_distances += d
        distances.append([d, reader[3]])
    distances.sort()
    distances = distances[:depth]
    a = {}
    for i in range(depth):
        a[distances[i][1]] = [1 - distances[i][0] / sum_distances] + [0] * 10000
    readen = set()
    with open('new_ratings.csv', 'r', encoding='Windows-1251') as file:
        reader = csv.reader(file)
        for j, row in enumerate(reader):
            if int(row[0]) == user_id:
                readen.add(int(row[1]))
            if a.get(int(row[0])):
                a[int(row[0])][int(row[1])] = int(row[2])
    recommendations = []
    k = -1
    for i in range(1, 10001):
        if i not in readen:
            recommendations.append([0, i])
            k += 1
            for key in a.keys():
                recommendations[k][0] += a[key][0] * a[key][i]
            recommendations[k][0] /= depth
    recommendations.sort(reverse=True)
    rec = []
    for i in recommendations:
        if i[0] != 0:
            rec.append(i[1])
    return rec


def print_book(row, message):
    keyboard = types.ReplyKeyboardMarkup(row_width=5)
    key1 = types.KeyboardButton(text='1')
    key2 = types.KeyboardButton(text='2')
    key3 = types.KeyboardButton(text='3')
    key4 = types.KeyboardButton(text='4')
    key5 = types.KeyboardButton(text='5')
    key_back = types.KeyboardButton(text='Назад')
    key_read = types.KeyboardButton(text='В список читаемого')
    key_skip = types.KeyboardButton(text='Пропустить')
    keyboard.add(key1, key2, key3, key4, key5)
    keyboard.add(key_read)
    keyboard.add(key_skip)
    keyboard.add(key_back)
    bot.send_message(message.chat.id,
                     "Название книги: *" + row[24] + '*\n' + 'Автор: ' + row[23] + '\n' +
                     'Год выпуска: ' + str(
                         int(float(row[8]))) + '\n' + 'Средняя оценка: ' +
                     row[12],
                     parse_mode="Markdown")
    bot.send_message(message.chat.id,
                     "Введите оценку книге по пятибалльной шкале (1-5) "
                     "или добавьте её в список читаемого:",
                     reply_markup=keyboard)


bot.enable_save_next_step_handlers(delay=1)
bot.load_next_step_handlers()

while True:
    try:
        bot.polling(none_stop=True, interval=0.2)
    except Exception as e:
        print(e)
        time.sleep(10)
