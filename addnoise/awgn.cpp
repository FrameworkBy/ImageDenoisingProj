#include "awgn.h"

#include <QDebug>

#define PI 3.1415926536

float AWGN_generator() {
    return AWGN_generator(1);
}

float AWGN_generator(float std)
{/* Генерация аддитивного белого гауссовского шума с нулевым средним и стандартным отклонением, равным 1. */

    float temp1;
    float temp2;
    float result;
    int p;

    p = 1;

    while( p > 0 )
    {
        temp2 = ( rand() / ( (float)RAND_MAX ) ); /* функция rand() генерирует
                                                       целое число между 0 и  RAND_MAX,
                                                       которое определено в stdlib.h.
                                                   */

        if ( temp2 == 0 )
        {// temp2 >= (RAND_MAX / 2)
            p = 1;
        }// конец if
        else
        {// temp2 < (RAND_MAX / 2)
            p = -1;
        }// конец else

    }// конец while()

    temp1 = cos( ( 2.0 * (float)PI ) * rand() / ( (float)RAND_MAX ) );
    result = sqrt( -2.0 * log( temp2 ) ) * temp1;

    return result*std;        // возвращаем сгенерированный сэмпл

}// конец AWGN_generator()
