#ifndef _APP_H_
#define _APP_H_


#include <time.h>
#include <sys/time.h>
/**
 * \brief Estrutura para tratar cronometro.
 */
struct stStopwatch_Unix
{
  struct timeval mStartTime;
  struct timeval mEndTime;
  double mCPUFreq;
  double mElapsedTime;
};
typedef struct stStopwatch_Unix Stopwatch;

/**
 * \brief Inicializa cronometro.
 */
#define FREQUENCY( prm ) {                                                    \
  prm.mCPUFreq = 0.0f;                                                       \
}


/**
 * \brief Dispara cronometro.
 */
#define START_STOPWATCH( prm ) {                                               \
   gettimeofday( &prm.mStartTime, 0);                                   \
}

/**
 * \brief Para cronometro.
 * retorna em segundos
 * para retornar em ms basta remover a linha 42 :)
 */
#define STOP_STOPWATCH( prm ) {                                                     \
  gettimeofday( &prm.mEndTime, 0);                                                     \
  prm.mElapsedTime = (1000.0f * ( prm.mEndTime.tv_sec - prm.mStartTime.tv_sec) + (0.001f * (prm.mEndTime.tv_usec - prm.mStartTime.tv_usec)) );                                                \
  prm.mElapsedTime /= 1000.0f; \
}

#endif

