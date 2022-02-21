# FindObjectGame
Find object game.

# Как запустить игру

# Пример скачивания репозитория с помощью git
cd D:/Test
git clone https://github.com/smt005/FindObjectGame.git
cd D:/Test/FindObjectGame
git submodule update --init --recursive

# Генерация проекта с помощью CMake
Запустить скрипт для соответствующей студии из папки "D:\Projects\Game\FindObjectGame\Source\Engine\Help\Script"
# WARNING
x64 вроде не рабтает.
Точно работает Win32, debug.

#WARNING
Есть ошибка с PhysX при генерации проекта. НУЖНО ПРОСТО ПОВТОРНО ЗАПУСТИТЬ СКРИПТ.

# Движокб "Engine"
Для создание нового проекта, достаточно подключить подмодуль Engine.
Исходники движка будут в папке Source.         Например, "D:\Projects\Game\FindObjectGame\Source\Engine".
Рядом нужно дабавить папку с игрой. Пример расположения, "D:\Projects\Game\FindObjectGame\Source\Game".

Минимальный шаблон игры и пример игры (с PhysX), вместе со скриптом CMake находятся в папках:
"D:\Projects\Game\FindObjectGame\Source\Engine\Help\Template project"
"D:\Projects\Game\FindObjectGame\Source\Engine\Help\Exampla project"

Нужно просто скопировать содержимое в папку "Source".