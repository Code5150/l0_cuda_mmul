# L0 Matrix Multiplication
В данной лабораторной работе мною были реализованы функции параллельного умножения матриц на GPU и на CPU. 
В реализации на GPU использовались функции библиотеки cuBLAS cublasSgemm (для параллельного умножения матриц) и cublasSgeam (для транспонирования матриц).
В реализации на CPU использовалась библиотека OpenMP, с помощью директив которой был распараллелен цикл матричного умножения.

В таблице 1 приведены сравнительные результаты времени умножения на CPU и GPU.

## Таблица 1. Ускорение
|                  |    100    |    200    |     400    |     600    |     800    |    1000    |    1200    |    1400    |    1600    |    1800    |    2000    |
|:----------------:|:---------:|:---------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
|  Время на CPU, с |  0,001382 |  0,004821 |  0,032684  |  0,063822  |  0,263463  |  0,518380  |  0,926440  |  1,429446  |  2,159996  |  3,127602  |  4,301015  |
| Время на GPU, мс |  0,023552 |  0,048704 |  0,175040  |  0,412288  |  0,904832  |  1,396608  |  2,484224  |  3,556960  |  5,494272  |  7,975904  |  10,208256 |
|  Ускорение, раз  | 58,678668 | 98,985710 | 186,723035 | 154,799558 | 291,173389 | 371,170722 | 372,929333 | 401,872948 | 393,135979 | 392,131350 | 421,327110 |

Таким образом, мы можем сделать вывод, что для параллельного умножения матриц больших размеров GPU подходит лучше, чем CPU, 
так как обеспечивает ускорение от 50 до 400 раз.
