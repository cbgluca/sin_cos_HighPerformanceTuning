#include "lib.h"

double my_sin(double);

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}


int main(){
    
    // printf("Insert value for sin: ");
    //double x;
    // scanf("%lf", &x);
    // double result = sinf(x);
    // printf("real sin(%lf) = %lf\n", x, result);


    // double result_approx = my_sin(x);
    // printf("approximate sin(%lf) = %lf\n", x, result_approx);


    int N = 100000; // 10 milioni di test
    printf("Generazione di %d valori casuali in corso...\n", N);

    // Alloco memoria per gli array
    double *in = malloc(N * sizeof(double));
    double *out_std = malloc(N * sizeof(double));
    double *out_my = malloc(N * sizeof(double));

    // Genero numeri casuali in un range più ampio (es. tra -10*PI e +10*PI)
    // per stressare la tua riduzione dell'intervallo (i cicli while)
    for (int i = 0; i < N; i++) {
        in[i] = ((double)rand() / RAND_MAX) * 20.0 * M_PI - 10.0 * M_PI;
    }

    // --- TEST LIBRERIA STANDARD ---
    double start_std = get_time();
    for (int i = 0; i < N; i++) {
        out_std[i] = sin(in[i]); // Nota: sin() per double, sinf() per float
    }
    double end_std = get_time();
    double time_std = end_std - start_std;

    // --- TEST TUA FUNZIONE ---
    double start_my = get_time();
    for (int i = 0; i < N; i++) {
        out_my[i] = my_sin(in[i]);
    }
    double end_my = get_time();
    double time_my = end_my - start_my;

    // --- CALCOLO ERRORE MASSIMO ---
    double max_error = 0.0;
    for (int i = 0; i < N; i++) {
        double error = fabs(out_std[i] - out_my[i]);
        if (error > max_error) {
            max_error = error;
        }
    }

    // --- STAMPA RISULTATI ---
    printf("\n=== RISULTATI BENCHMARK ===\n");
    printf("Tempo Standard (math.h): %f secondi\n", time_std);
    printf("Tempo my_sin           : %f secondi\n", time_my);
    
    if (time_my < time_std) {
        printf("Performance            : La tua è più VELOCE di %.2fx!\n", time_std / time_my);
    } else {
        printf("Performance            : La tua è più LENTA di %.2fx\n", time_my / time_std);
    }
    
    printf("Errore Massimo         : %e\n", max_error);

    free(in); free(out_std); free(out_my);
    return 0;
}



double my_sin(double x){
    //bringing it ot the right interval [-pi, +pi]
    double test, x_2,x_3,x_5,x_7,x_9;
    int f_3,f_5,f_7,f_9;

    const double inv_pi = 1.0 / M_PI;

    int k = round(x * inv_pi);

    x = x - (k * M_PI);


    // while(x > M_PI){
    //     x -= pi_2;
    // }
    // while(x < -M_PI){
    //     x += pi_2;
    // }

    //printf("\ntest x:%lf\n", x);

    x_2= x*x;
    x_3= x_2*x;
    x_5= x_3*x_2;
    x_7= x_5*x_2;
    x_9= x_7*x_2;

    f_3 = 6;
    f_5 = 120;
    f_7 = 5040;
    f_9 = 362880;

    test = x - x_3 * (1.0/f_3) + x_5 * (1.0/f_5) - x_7 * (1.0/f_7) + x_9 * (1.0/f_9);
    //test = x - x_3 * (1.0/f_3) + x_5 * (1.0/f_5);

    //printf("con 1/f: %lf\n", test);
    //x = x - x_3 / f_3 + x_5 / f_5 - x_7 / f_7 + x_9 / f_9;
    return (k & 1) ? -test : test;
}


// double power (double num, int exp){
//     for(int i = 1; i<exp; i++){
//         num *= num;
//     }
//     return num;
// }

