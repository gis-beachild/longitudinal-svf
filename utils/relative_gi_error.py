"""Calcule la moyenne ± écart-type échantillonnal des erreurs relatives GI."""

from statistics import mean, stdev

GT = {
    22: 1.0248444,
    23: 1.0068339,
    24: 1.0669769,
    25: 1.0813942,
    26: 1.0938596,
    27: 1.1260786,
    28: 1.1629825,
    29: 1.2016829,
    30: 1.217135,
    31: 1.2668083,
    32: 1.3104898,
    33: 1.3306319,
    34: 1.3684465,
    35: 1.4374748,
    36: 1.4428234,
}

GI = {
    22: 1.0124567,
    23: 1.0318758,
    24: 1.0478618,
    25: 1.0525575,
    26: 1.0699219,
    27: 1.0906718,
    28: 1.1137992,
    29: 1.1362314,
    30: 1.1646326,
    31: 1.195259,
    32: 1.2314886,
    33: 1.2745699,
    34: 1.3200914,
    35: 1.3715373,
    36: 1.4308581,
}




def relative_gi_error(gt, gi):
    """Retourne (moyenne, SD avec ddof=1), en associant les valeurs par âge."""
    if not gt or gt.keys() != gi.keys():
        raise ValueError('GT et GI doivent contenir les mêmes âges et être non vides.')
    if any(value <= 0 for value in gt.values()):
        raise ValueError('Les valeurs GT doivent être strictement positives.')
    if len(gt) < 2:
        raise ValueError('Au moins deux points sont nécessaires pour calculer la SD.')
    errors = [abs(gi[age]) / value for age, value in gt.items()]
    return mean(errors), stdev(errors)


if __name__ == '__main__':
    print('Âge\tErreur relative')
    for age in sorted(GT):
        print(f'{age}\t{GI[age] / GT[age]:.8f}')
    error, sd = relative_gi_error(GT, GI)
    print(f'\nRel. GI Err. (N={len(GT)}, mean ± SD, ddof=1) = '
          f'{error:.8f} ± {sd:.8f} '
          f'({100 * error:.2f} \\pm {100 * sd:.2f} %)')
