# Метрики и последовательность эксперимента по оценке перевода WTQ

Описание составлено по текущей реализации `evaluate.py`, `tables.py`, `mentions.py`, `metrics.py`, `loaders.py` и `llm_judge.py`. Значения в таблицах приведены из предоставленного отчёта на 90 примерах. Промпты в конце извлечены из текущего кода; их соответствие именно этому запуску нужно подтвердить по `rubric_hash` в его результатах.

## 1. Что оценивает эксперимент

Объект оценки — готовый перевод WikiTableQuestions с английского на русский: вопросы, таблицы и связи между упоминаниями в вопросах и элементами таблиц. Единица примера — вопрос с соответствующей таблицей и trace-файлом перевода. Разные примеры могут ссылаться на одну исходную таблицу.

Python проверяет формальные свойства и рассчитывает агрегаты. Модель-судья выносит отдельные бинарные смысловые решения. Эксперимент не выполняет перевод, не решает вопросы и не использует эталонные ответы `targetValue`.

## 2. Основные метрики

Для положительных критериев `1` означает успешную проверку, `0` — проблему. Итоговый процент — доля единиц среди доступных оценок. Для критических флагов направление обратное: `1` означает обнаруженную ошибку, поэтому меньше — лучше. Критерии независимы по заданию судье, но фактически могут пересекаться на одних примерах.

### 2.1. Перевод вопросов

| Метрика отчёта / ключ summary | Результат | Что измеряет и как трактовать | Кто оценивает |
|---|---:|---|---|
| Сохранение смысла — Meaning preservation / `meaning_preservation` | 82,2% | Долю вопросов, где сохранены предмет вопроса, действие, отношения, условия, временные ограничения и существенная информация. Ошибка — существенный пропуск, искажение или выдуманное условие. Например, исчезновение ограничения «после 2000 года». | LLM, поле примера `meaning_preserved` |
| Правильность ключевых элементов — Key elements correctness / `key_elements_correctness` | 76,7% | Долю вопросов без существенных ошибок в именах, организациях, географии, датах, числах, единицах и специальных терминах. Транслитерация, общепринятый перевод и латиница допустимы, если сущность та же. При отсутствии таких элементов судье предписано поставить 1. | LLM, `key_elements_correct` |
| Сохранение логики — Logic preservation / `logic_preservation` | 96,7% | Долю вопросов с сохранёнными операциями и отношениями: больше/меньше, до/после, первый/последний, отрицание, и/или, количество, сумма, среднее, максимум, минимум, разность. Замена «больше всего» на «меньше всего» — ошибка. | LLM, `logic_preserved` |
| Естественность русского языка — Russian naturalness / `russian_naturalness` | 60,0% | Долю вопросов с естественной лексикой, грамматикой, согласованием, синтаксисом и порядком слов, без заметных калек. Сохранённый смысл не гарантирует естественность. Регистр и конечный вопросительный знак не штрафуются, если следуют стилю оригинала. | LLM |
| Доля критических ошибок — Critical error rate / `critical_error_rate` | 26,7% | Долю примеров с итоговым критическим флагом: ошибка может изменить смысл или правильный ответ. Это объединение источников ниже; не фактически измеренная доля неправильных ответов. Чисто стилистическая проблема не должна сама по себе считаться критической. | Python объединяет флаги, поле примера `critical_error` |
| Критические ошибки по решению судьи — `critical_error_breakdown.llm` | 18,9% | Долю вопросов, где модель непосредственно обнаружила потенциально существенную ошибку: неверную сущность, число, дату, операцию или потерю условия. | LLM, `critical_error_llm` |
| Критические ошибки из упоминаний — `critical_error_breakdown.from_mentions` | 11,1% | Долю примеров, где нарушено сохранение набора масок, осталось `<TABLE_REF…>` либо есть упоминание с `correctness=0`. Включает смысловые решения модели, поэтому не является полностью детерминированным источником. | Python и результаты проверки упоминаний, `critical_error_from_mentions` |
| Критические ошибки из значений таблицы — `critical_error_breakdown.from_table_values` | 1,1% | Долю примеров, где связанное с упоминанием неизменяемое значение не прошло ни строгую, ни мягкую проверку. Изменения других ячеек не создают этот флаг автоматически. | Python, `critical_error_from_table` |

Источники критических ошибок могут пересекаться. Их проценты нельзя складывать. Хорошая логика вопроса также не гарантирует правильность названной сущности или величины.

### 2.2. Перевод таблиц

| Метрика отчёта / ключ summary | Результат | Что измеряет и как трактовать | Кто оценивает |
|---|---:|---|---|
| Сохранение структуры — Structure preservation / `structure_preservation` | 91,1% | Долю примеров с совпадающим количеством строк данных, количеством заголовков столбцов и длинами соответствующих строк. Проверка размеров не обнаруживает все перестановки строк или столбцов при неизменных размерах. | Python, поле примера `structure_preserved` |
| Правильность ячеек — Cell correctness / `cell_correctness` | 82,6% | Долю оценённых текстовых ячеек и заголовков, верно передающих содержание оригинала. Искажение значения — ошибка; допустимые варианты передачи имён и несущественная стилистическая вариативность не штрафуются. Это доля ячеек, а не полностью правильных таблиц. | LLM; пропущенный перевод и заполнение пустой исходной ячейки проверяет Python |
| Сохранение информации — Information preservation / `information_preservation` | 90,5% | Долю текстовых ячеек без потерь исходных сведений: элементов списка, уточнений, примечаний, частей в скобках, единиц. «1st, Western» → «1-е место» не проходит из-за потери Western. | LLM и правила для пустых/отсутствующих переводов |
| Отсутствие добавленной информации — No added information / `no_added_information` | 96,2% | Долю текстовых ячеек без новых фактов и пояснений, которых не было в оригинале. Грамматически необходимые элементы русского не считаются добавлением. «1st» → «1-е место в Западном дивизионе» не проходит. Пустой перевод может пройти этот критерий, одновременно провалив сохранение информации. | LLM и правила для пустых ячеек |
| Точность неизменяемых значений — Immutable accuracy / `immutable_accuracy` | 93,0% | Долю отобранных ячеек, классифицированных как числа, символы, даты, коды, URL или email, прошедших строгую проверку. Обычно требуется совпадение нормализованных при загрузке строк; для распознанных дат с названиями месяцев сравнивается разобранное значение. | Python |
| Точность с допуском оформления — Immutable accuracy, lenient formatting / `immutable_accuracy_lenient` | 93,9% | Мягкий вариант: для числовых ячеек сравниваются извлечённые числа после эвристической нормализации разделителей; для символов допускаются некоторые варианты тире и пробелы. Для дат остаётся сравнение разобранной даты, для URL и кодов — строгое сравнение. Например, `9,734` и `9 734` могут совпасть. Это эвристика, а не полноценная проверка всех знаков, единиц и числовой семантики. | Python |

Первые три метрики ячеек не включают неизменяемые ячейки: для них есть отдельные показатели. Полностью пустые пары ячеек исключаются. Пустая исходная ячейка с добавленным переводом входит в текстовые проверки как ошибка правильности и добавление информации.

### 2.3. Упоминания таблицы в вопросе

Упоминание — фрагмент вопроса, связанный с заголовком или значением таблицы. Например, «в каком году» связано с заголовком Year / «Год».

| Метрика отчёта / ключ summary | Результат | Что измеряет и как трактовать | Кто оценивает |
|---|---:|---|---|
| Сохранение упоминаний — Preservation / `preservation` | 100,0% | Долю упоминаний, чьи маски присутствуют в `forward_masks` trace. Измеряет техническое сохранение ссылки на отслеживаемом этапе, а не правильность восстановленного русского текста. | Python, поле элемента `preserved` |
| Все маски сохранены — all masks kept / `preservation_all_masks` | 100,0% | Долю примеров с доступной проверкой, где нет потерянных и добавленных масок. Наборы сравниваются как множества; кратность повторений этим сравнением не контролируется. | Python, поле раздела `preservation` |
| Правильность упоминаний — Correctness / `correctness` | 92,5% | Долю упоминаний, где цепочка «английское упоминание → исходный элемент → русский элемент → фрагмент вопроса» корректна. Русский фрагмент должен обозначать тот же элемент, а перевод самого элемента должен быть верным. | Обычно LLM; некоторые случаи решаются или переопределяются правилами |
| Смысловая согласованность — Semantic consistency / `semantic_consistency` | 90,2% | Долю упоминаний, где русский фрагмент вопроса и русский элемент таблицы совпадают по смыслу. Падежи, порядок слов, допустимые сокращения и варианты транслитерации могут различаться. Два одинаково неверных перевода могут быть согласованы между собой, поэтому этот критерий не заменяет правильность цепочки. | Обычно LLM; есть детерминированный успешный случай для неизменяемых значений |
| Поверхностное совпадение — Surface consistency / `surface_consistency` | 26,0% | Долю доступных пар «извлечённый фрагмент — текст элемента» с одинаковой нормализованной формой. Нормализация убирает различия регистра, ё/е, пунктуации и пробелов, но не приводит слова к начальной форме. «Год» и «году» не совпадают. Диагностика, а не основной критерий качества. | Python после извлечения фрагмента |
| Текст ячейки в вопросе — Cell text in question / `cell_text_in_question` | 27,6% | Долю упоминаний, где нормализованный текст русского элемента целиком найден в вопросе с учётом границ слов. Не требует совпадения с извлечённым моделью фрагментом. Низкое значение возможно при правильном склонении или перефразировании. | Python |

### 2.4. Дополнительные счётчики и локальные поля

| Поле | Значение |
|---|---|
| `table.cells_judged_by_llm` | Количество отобранных ячеек с полученным решением LLM; это не число уникальных API-оценок, поскольку одинаковые ячейки могут использовать одно решение. |
| `table.cells_llm_failed` | Количество ячеек, для которых пакетная оценка не дала результата. Такие смысловые оценки обычно остаются `null`. |
| `mentions.n_mentions` | Общее количество элементов упоминаний, включая элементы без доступной оценки. |
| `cells_total`, `cells_evaluated` | Число подготовленных пар ячеек и число выбранных для итоговой оценки. Полностью пустые пары в подготовленные оценки не входят. |
| `n_text_cells`, `n_immutable_cells` | Количество отобранных текстовых и неизменяемых ячеек внутри примера. Это не обязательно число доступных смысловых решений. |
| Локальные `cell_translation_accuracy`, `information_preservation`, `no_added_information`, `immutable_accuracy`, `immutable_accuracy_lenient` | Средние соответствующих доступных оценок внутри одного примера. |
| `translation_variants_consistent` | Локальный флаг: найдено не более одного уникального русского варианта для рассматриваемых повторов исходного значения. Это строковая проверка, не смысловая; при отсутствии вариантов также получается 1. В стандартный summary не агрегируется. |
| `issues`, `note`, `flags`, `warnings`, `section_errors` | Объяснения и сведения о проблемах; это не самостоятельные проценты качества. |

## 3. Диагностика исходного пайплайна

Оценщик копирует эти значения из trace, а не вычисляет заново. Точные модели эмбеддингов, настройки BERTScore, состав групп и способ сравнения вопросов следует искать в коде исходного переводчика или метаданных trace. По этому репозиторию нельзя достоверно восстановить эти настройки. В частности, нельзя без дополнительной проверки утверждать, что метрики вопроса сравнивают именно оригинал и обратный перевод.

| Метрика | Пояснение |
|---|---|
| `question_cosine_similarity` | Сходство представлений вопроса по cosine, рассчитанное исходным пайплайном. Высокое сходство не исключает замену сущности, числа или логического условия. |
| `question_bert_precision` | Компонент precision диагностической метрики BERTScore вопроса: семантическое соответствие сопоставляемых токенов на стороне кандидата относительно референса. Ориентацию сравнения задаёт исходный пайплайн. Не доля правильно переведённых слов. |
| `question_bert_recall` | Компонент recall BERTScore вопроса: покрытие токенов референса семантически соответствующими токенами кандидата. Не прямой процент сохранённой информации. |
| `question_bert_f1` | Комбинация precision и recall BERTScore вопроса, рассчитанная исходным пайплайном. Значение 0,805 не означает 80,5% правильных вопросов. |
| `back_table_groups_compared` | Число групп, сопоставленных при диагностике обратного перевода таблицы по similarity. Это объём сравнения, а не качество. |
| `back_table_mean_similarity` | Среднее сходство групп обратного перевода таблицы внутри примера. |
| `back_table_min_similarity` | Минимальное сходство группы внутри примера; помогает выделять слабые участки, скрытые общим средним. |
| `back_table_bert_groups_compared` | Количество групп, вошедших в BERTScore-диагностику обратного перевода таблицы. |
| `back_table_bert_mean_precision` | Средний precision BERTScore по сравниваемым группам внутри примера. |
| `back_table_bert_mean_recall` | Средний recall BERTScore по сравниваемым группам внутри примера. |
| `back_table_bert_mean_f1` | Средний F1 BERTScore по сравниваемым группам внутри примера. |
| `back_table_bert_min_precision` | Минимальный precision BERTScore среди групп примера. |
| `back_table_bert_min_recall` | Минимальный recall BERTScore среди групп примера. |
| `back_table_bert_min_f1` | Минимальный F1 BERTScore среди групп примера. |
| `n_ambiguous_mentions` | Число записей в `quality.ambiguous_mentions` trace. Ноль означает отсутствие зарегистрированных неоднозначностей, но не доказывает верность всех привязок. |
| `n_pipeline_errors` | Число записей в `quality.errors` исходного пайплайна. Не является числом технических ошибок текущего оценщика. |
| `n_pipeline_warnings` | Число записей в `quality.warnings` исходного пайплайна. Среднее число предупреждений не равно доле примеров с предупреждениями. |

Для каждой числовой диагностики summary хранит `n`, среднее, медиану, минимум и максимум, а также средние и количества отдельно для примеров с критическим флагом 0 и 1. Текстовый отчёт показывает только среднее, медиану и n. Поэтому `back_table_bert_min_f1 mean=0.774` — среднее минимумов по примерам, а не глобальный минимум запуска.

## 4. Как рассчитываются агрегаты

| Обозначение | Пояснение |
|---|---|
| `k` | Число единиц: успешных проверок для положительных критериев или ошибок для критических флагов. |
| `n` | Число доступных бинарных оценок конкретной метрики. Это могут быть вопросы, ячейки или упоминания. |
| `rate` | `k / n`; в отчёте умножается на 100%. |
| `macro.rate`, per-example mean | Среднее долей внутри примеров. Каждый пример с доступными оценками имеет одинаковый вес. |
| `macro.n_examples` | Число примеров, участвующих в этом среднем. |
| `ci95` | Интервал Уилсона с z=1,96 для общей бинарной доли. Для macro отдельный интервал кодом не рассчитывается. |
| `null`, n/a | Оценка отсутствует. Исключается из знаменателя, а не считается успехом или ошибкой. |

Общая доля по ячейкам сильнее зависит от примеров с большим числом оценённых ячеек. Например, 82,6% общей правильности и 84,2% среднего по примерам — разные способы взвешивания одних данных.

Интервал Уилсона не учитывает ошибки модели-судьи и зависимость ячеек внутри таблиц. Повторное использование таблиц также создаёт зависимость наблюдений. Поэтому узкие интервалы по тысячам ячеек не следует считать полной оценкой неопределённости качества датасета. Единого интегрального балла в проекте нет.

## 5. Полный pipeline эксперимента

### Этап 1. Подготовка входов

Нужны `dataset/training.tsv` с ID, русским вопросом и путём к русской таблице; исходные английские таблицы; русские таблицы; trace-файл для каждого ID. Trace содержит английский вопрос, путь `source_context`, упоминания, координаты, маски и диагностику исходного перевода.

Перевод, выделение исходных упоминаний, маскирование и обратный перевод выполнены до запуска оценщика. Этот репозиторий использует их результаты. Порядок внутренних шагов исходного переводчика здесь не реализован.

### Этап 2. Настройка запуска и рубрики

Загружаются параметры CLI и `.env`: пути, провайдер, модель, температура, seed, размер выборки, число работников, лимит ячеек и размер пакета. По умолчанию: 200 примеров, seed=42, 4 работника, 150 ячеек, пакет ячеек 40, температура 0. Это значения по умолчанию, а не подтверждённые параметры предоставленного запуска.

Хеш четырёх текстовых шаблонов из `llm_judge.py` сверяется с `rubric_lock.json`. При изменении промптов под уже зарегистрированной версией запуск останавливается и требует новой версии рубрики. Температура 0 не гарантирует абсолютной воспроизводимости внешнего сервиса; при неподдерживаемом параметре клиент может повторить вызов без него.

### Этап 3. Формирование выборки

ID приводятся к естественному порядку и отбираются псевдослучайно по seed. При размере не больше нуля или не меньше датасета берутся все ID. Альтернатива — явный список `--ids-file`, проверяемый на наличие ID в датасете. Выборка сохраняется в `sample_ids.json`.

### Этап 4. Продолжение или новый запуск

При существующем файле результатов требуется `--resume` либо `--force`. В режиме resume проверяется хеш рубрики и переиспользуются записи выбранных ID без `section_errors`. Проверка модели и всех остальных параметров прежнего запуска при этом не выполняется: совместимость условий должен обеспечить экспериментатор. Некоторые локальные сбои LLM не создают `section_errors`, поэтому наличие записи без ошибок раздела не гарантирует полного покрытия.

### Этап 5. Загрузка и нормализация каждого примера

Загружаются trace, английская таблица по `source_context` и русская таблица по `context` датасета. Английский вопрос берётся из trace, русский — из `utterance` датасета. Если русский вопрос trace отличается, добавляется предупреждение, а оценивается вариант датасета.

При чтении таблиц нормализуются пробелы, переносы и некоторые escape-последовательности. Заголовок хранится отдельно; строки данных имеют индексы с нуля. Пары ячеек формируются по одинаковым координатам, без смыслового выравнивания.

### Этап 6. Проверка структуры и подготовка ячеек

Python сравнивает размеры и длины строк, определяет отсутствующие соответствия. Исходные ячейки классифицируются как пустые, текстовые, числовые, даты, символы, коды, URL/email. Полностью пустые пары исключаются. Отсутствующий перевод текста и добавление текста в пустую исходную ячейку оцениваются правилами. Неизменяемые значения получают строгую и мягкую оценки.

Проверка структуры охватывает таблицу целиком. Индивидуальная оценка содержания после отбора охватывает только выбранные ячейки. Добавленные строки и столбцы выявляются структурой; для них нет отдельного полного обхода текстовых ячеек без исходной пары.

### Этап 7. Отбор ячеек

При действующем лимите обязательно включаются подготовленные ячейки заголовков и все ячейки, связанные с упоминаниями, включая координаты-кандидаты. Остальные добираются воспроизводимо с seed на основе seed запуска и ID. Если обязательных ячеек больше лимита, они остаются все. `--max-cells 0` отключает ограничение. Лимит относится ко всем выбранным типам ячеек, а не только к вызовам LLM.

### Этап 8. Смысловая оценка текстовых ячеек

Непустые текстовые пары отправляются судье с доменом и английским заголовком столбца. Внутри примера одинаковые тройки «заголовок, оригинал, перевод» оцениваются один раз, затем решение распространяется на соответствующие ячейки. Каждая ячейка при агрегировании остаётся отдельным наблюдением.

Модель возвращает `correct`, `information_preserved`, `no_added_information` и короткое замечание. Полная таблица и вопрос в этой задаче судье не передаются.

### Этап 9. Оценка упоминаний

Из trace сравниваются исходные и forward-маски. Для упоминания выбирается явная координата либо первая из списка кандидатов, извлекаются исходный и русский элементы, собираются варианты перевода повторов.

Отсутствующая исходная координата помечается как проблема данных; она сама по себе не объявляется ошибкой перевода. Отсутствующая русская ячейка даёт `correctness=0`. Если неизменяемая ячейка прошла строгую проверку и её текст найден в вопросе, правильность и согласованность могут быть установлены правилом без LLM.

В остальных случаях модель получает оба вопроса и данные упоминаний, извлекает русский фрагмент и возвращает `correctness`, `semantic_consistency`, `note`. Пакет упоминаний по умолчанию содержит до 25 элементов. После ответа Python принудительно обнуляет правильность при связанном неизменяемом значении, не прошедшем обе проверки, проверяет наличие извлечённого фрагмента и рассчитывает поверхностные признаки. Отбрасывание ненайденного фрагмента само по себе не пересчитывает все положительные решения модели — такие записи полезно проверять вручную.

### Этап 10. Оценка вопроса

В отдельном вызове модель получает домен, английский вопрос и русский вопрос. Она оценивает четыре критерия вопроса, критическую ошибку и список `issues`. Саму таблицу, эталонный ответ и готовые оценки ячеек этот вызов не получает. Промпт запрещает решать вопрос или оценивать его ответимость.

### Этап 11. Формирование итогового критического флага

Python объединяет три источника: решение судьи вопроса, флаг из упоминаний и флаг из связанных значений таблицы. Если хотя бы один источник равен 1, итог равен 1. Если все три равны 0, итог равен 0. Иначе итог остаётся `null`.

В текущей реализации отсутствие оценки отдельного упоминания не обязательно делает весь источник `from_mentions` неопределённым: при доступном разделе и отсутствии явных триггеров он может стать 0. Поэтому итоговый ноль не доказывает полноту проверки всех упоминаний.

### Этап 12. Кэш, валидация ответов и обработка сбоев

Каждый LLM-запрос сначала проверяет кэш. Ключ включает провайдера, модель, заданную температуру, хеш рубрики и тексты системного и пользовательского сообщений. Ответ валидируется как JSON с нужными ID и бинарными полями.

При ошибочном формате, обрезанном ответе или некоторых временных сбоях выполняются повторы; по умолчанию предусмотрены до 5 попыток на задачу. При устойчивом сбое пакета ячеек или упоминаний он рекурсивно делится пополам до отдельного элемента. Неуспешный элемент остаётся без смысловой оценки. Ошибки разделов сохраняются отдельно; сбой загрузки может исключить пример из `evaluation.jsonl` и попасть в `errors.jsonl`.

Примеры обрабатываются параллельно, но внутри одного примера порядок разделов — таблица, упоминания, вопрос, итоговый критический флаг.

### Этап 13. Агрегация и сохранение

Успешно сформированные записи сортируются по ID. Python рассчитывает доли, интервалы Уилсона, средние по примерам и статистику диагностики. `null` исключается отдельно для каждой метрики, поэтому знаменатели могут отличаться.

| Артефакт | Содержание |
|---|---|
| `evaluation.jsonl` | Одна запись на сформированный пример: исходные тексты, оценки, ячейки, упоминания, замечания и ошибки разделов. |
| `summary.json` | Агрегаты и метаданные: число выбранных и оценённых примеров, модель, версия/хеш рубрики и другие параметры. |
| `report.txt` | Человекочитаемая сводка. |
| `errors.jsonl` | Ошибки загрузки и разделов оценщика. Не список смысловых ошибок перевода. |
| `sample_ids.json` | Зафиксированные ID выборки и параметры отбора. |
| `cache/` | Валидированные JSON-ответы судьи. |

`--no-llm` оставляет только проверки правил; это неполная смысловая оценка. `--aggregate-only` пересчитывает сводку из существующего JSONL и печатает её, без новых вызовов судьи; текущая реализация этого режима не перезаписывает `summary.json` и `report.txt` и не восстанавливает полные метаданные запуска.

### Этап 14. Проверка результатов исследователем

Это отдельный методический этап, а не автоматическая часть кода: проверить покрытие и ошибки, вручную разобрать критические примеры и случайную часть примеров без флага, подтвердить корректность координат и оценить согласие судьи с человеком. Для сравнения версий фиксируются ID, входные данные, модель, рубрика и параметры отбора/чтения таблиц. По одному отсутствию критического флага нельзя заключать, что пример полностью правильный.

## 6. Промпты, задающие роль судьи

В проекте используются **три отдельных системных промпта**: для вопросов, ячеек и упоминаний. Ниже приведены полные оригинальные тексты из текущего `llm_judge.py`, без перевода и сокращений. У вопросов также приведён точный шаблон пользовательского сообщения. Для ячеек и упоминаний пользовательское сообщение — JSON с полями, описанными в соответствующем системном промпте; неизвестный домен передаётся как `unknown`.

### 6.1. Системный промпт оценки вопросов

```text
You are an expert English-Russian translation reviewer. You evaluate the Russian translation of a question from the WikiTableQuestions (WTQ) dataset. WTQ questions are asked about Wikipedia tables and are answered by reasoning over the table (lookup, comparison, counting, sums, averages, superlatives, ordering, differences, etc.).

You judge ONLY the translation against the English source. Do NOT try to answer the question and do NOT judge whether it is answerable.

Give four independent binary verdicts (1 = OK, 0 = problem) and one critical-error flag.

1. meaning_preserved
   1 = the Russian question asks about the same thing as the English one: same object of the question, action, relations, conditions, scope, time reference and all essential information.
   0 = something essential is distorted, missing or invented.

2. key_elements_correct
   1 = all key elements are translated correctly: person names, organisations, league/competition names, geographic names, dates, numbers, units, domain-specific terms, other entities. If the question contains no such elements, answer 1.
   0 = at least one substantive error in such an element (wrong entity, wrong number or date, wrong unit, mistranslated term, a name garbled beyond recognition).
   Names may be transliterated, translated with an established Russian equivalent, or left in Latin script; any of these is fine when the entity stays clearly the same. Normal Russian inflection of names and numerals is expected and is not an error.

3. logic_preserved
   1 = the logic is fully preserved, including: before/after, more/less, highest/lowest, first/last, at least/at most, negation, conditions, and/or, count, sum, average, max, min, difference, ranking and any other operation or relation that affects the answer.
   0 = any of these has been changed, weakened, lost or added.
   Example: "What team scored the most goals?" -> "Какая команда забила меньше всего голов?" is logic_preserved = 0.

4. russian_naturalness
   1 = natural, grammatical Russian that a native speaker would accept as a normal question: correct grammar and agreement, syntax, word order, lexis; no calques; easy to read.
   0 = a noticeable problem (ungrammatical, clearly unnatural word order, calque, wrong agreement, awkward or misleading wording).
   Do NOT penalise capitalisation or the presence/absence of a final question mark when it simply follows the style of the source.

5. critical_error
   1 = there is an error that could plausibly change the meaning or the correct answer, e.g. first<->last, more<->less, highest<->lowest, before<->after, a wrong aggregation, a wrong number, a wrong date, a wrong entity, an essential condition lost, or a table-related value distorted. Pure style problems are never critical.
   0 = otherwise.
   If logic_preserved = 0 because of a change that affects the answer, critical_error must be 1.

Judge every criterion independently: do not lower one criterion because another one failed, and do not give 0 for a harmless paraphrase.

Return ONLY a JSON object (no markdown, no commentary) in exactly this form:
{"issues": ["<very short description of each real problem; empty list if none>"], "meaning_preserved": 0 or 1, "key_elements_correct": 0 or 1, "logic_preserved": 0 or 1, "russian_naturalness": 0 or 1, "critical_error": 0 or 1}
```

### 6.2. Пользовательский шаблон оценки вопросов

```text
Domain of the table: {domain}

English source question:
<<<
{src}
>>>

Russian translation:
<<<
{tr}
>>>
```

### 6.3. Системный промпт оценки ячеек

```text
You are an expert English-Russian translation reviewer. You evaluate the Russian translation of individual cells of a table from the WikiTableQuestions dataset (the English table was translated cell by cell into Russian).

You receive a JSON object with the table domain and a list of cells. Each cell has: "i" (id), "column" (the English header of its column, for context only), "source" (English cell text) and "translation" (Russian cell text). Judge every cell independently.

For each cell give three binary verdicts (1 = OK, 0 = problem):

- correct: the Russian text is a faithful translation of the English cell: it conveys the same meaning and nothing in it is mistranslated or distorted. Person, team and place names may be transliterated, translated with an established Russian equivalent, or kept in Latin script - all fine if clearly the same entity. Standard abbreviations may be kept or localised. Do not penalise stylistic variation.
- information_preserved: every piece of information of the English cell is still present in the Russian cell (no dropped list items, qualifiers, notes, parenthesised parts, units, etc.).
- no_added_information: the Russian cell contains nothing that is absent from the English cell (no added words that carry extra facts, qualifiers, divisions, units or explanations). Grammatical necessities of Russian (case endings, "г." after a year, etc.) are not added information.

Examples: English "1st, Western" -> Russian "1-е место": information_preserved = 0 ("Western" is lost). English "1st" -> Russian "1-е место в Западном дивизионе": no_added_information = 0.

Judge strictly the cell against its source; ignore any question the table may be used for.

Return ONLY a JSON object (no markdown):
{"results": [{"i": <id>, "correct": 0 or 1, "information_preserved": 0 or 1, "no_added_information": 0 or 1, "note": "<short reason if any verdict is 0, otherwise empty string>"}, ...]}
Include exactly one result per input cell.
```

### 6.4. Системный промпт оценки упоминаний

```text
You are an expert English-Russian translation reviewer. In the WikiTableQuestions dataset a question mentions elements of a table (column headers or cell values). Both the question and the table were translated from English to Russian. You check whether each table mention survived the translation correctly.

Input is a JSON object with:
- "source_question" and "translated_question";
- "mentions": a list; each mention has "id", "source_mention" (the wording in the English question), "source_element_type" ("header" = column header, "cell" = cell value), "source_table_text" (the English header/cell the mention refers to), "translated_table_text" (its translation in the Russian table) and optionally "other_translated_table_texts" (translations of other cells holding the same English value).

For every mention produce:

1. span: copy, character by character, the shortest contiguous fragment of "translated_question" that renders "source_mention" in Russian (keep the inflected form used in the question). Include only words that translate words of source_mention; do not include extra neighbouring words. If the translated question has no counterpart of the mention, use "".

2. correctness (chain check): source_mention -> source_table_text -> translated_table_text -> span.
   1 = the span refers to the same table element (same entity / value / column) as the English mention, AND translated_table_text is a correct translation of source_table_text.
   0 = the span is missing, or refers to a different entity/value/column, or translated_table_text is not a correct translation of source_table_text.

3. semantic_consistency: compare ONLY the span with translated_table_text (the Russian side).
   1 = they mean the same thing, even if the form differs: inflection/case, word order, capitalisation, transliteration variant, abbreviation vs full form, a harmless generic word added or dropped. Example: table "Премьер-лига" and span "в английской Премьер-лиге" are consistent.
   0 = they denote different entities/values/columns or different meanings, or there is no span.

Return ONLY a JSON object (no markdown):
{"results": [{"id": "<mention id>", "span": "<fragment or empty string>", "correctness": 0 or 1, "semantic_consistency": 0 or 1, "note": "<short reason if any verdict is 0, otherwise empty string>"}, ...]}
Include exactly one result per input mention.
```

