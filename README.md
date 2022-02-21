# Find object game.

# Как запустить игру

# Пример скачивания репозитория с помощью git
cd D:/Test

git clone https://github.com/smt005/FindObjectGame.git

cd D:/Test/FindObjectGame

git submodule update --init --recursive

# Генерация проекта с помощью CMake
Запустить скрипт для соответствующей студии из папки "D:\Projects\Game\FindObjectGame\Source\Engine\Help\Script"
# WARNING
- x64 вроде не рабтает. Точно работает Win32, debug.
- Есть ошибка с PhysX при генерации проекта. НУЖНО ПРОСТО ПОВТОРНО ЗАПУСТИТЬ СКРИПТ.

# Движокб "Engine"
Для создание нового проекта, достаточно подключить подмодуль Engine.
Исходники движка будут в папке Source.
Рядом нужно дабавить папку с игрой.

"D:\Projects\Game\FindObjectGame\Source\Engine", пример расположения движка
"D:\Projects\Game\FindObjectGame\Source\Game", пример расположения игры

Минимальный шаблон игры и пример игры (с PhysX), вместе со скриптом CMake находятся в папках:
"D:\Projects\Game\FindObjectGame\Source\Engine\Help\Template project"
"D:\Projects\Game\FindObjectGame\Source\Engine\Help\Exampla project"

Нужно просто скопировать содержимое в папку "Source".