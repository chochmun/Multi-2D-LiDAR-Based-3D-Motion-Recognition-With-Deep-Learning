import re
def calculate_average(input_data):
    # Filter lines that don't end with 'step' and only keep decimal values
    float_numbers = [float(line) for line in input_data.splitlines() if re.match(r'^\d+\.\d+$', line)]
    
    if not float_numbers:  # Return None if there are no numbers
        return None
    
    # Calculate and return the average of the extracted numbers
    return sum(float_numbers) / len(float_numbers)

# Example input data
input_data = """
0.07799983024597168
1/1 [==============================] - 0s 26ms/step
0.07300162315368652
1/1 [==============================] - 0s 21ms/step
0.06100010871887207
1/1 [==============================] - 0s 24ms/step
0.0670003890991211
1/1 [==============================] - 0s 21ms/step
0.05700206756591797
1/1 [==============================] - 0s 71ms/step
0.15152978897094727
1/1 [==============================] - 0s 38ms/step
0.11099982261657715
1/1 [==============================] - 0s 28ms/step
0.06600189208984375
1/1 [==============================] - 0s 20ms/step
0.062001943588256836
1/1 [==============================] - 0s 24ms/step
0.08000016212463379
1/1 [==============================] - 0s 23ms/step
0.06399703025817871
1/1 [==============================] - 0s 23ms/step
0.07800078392028809
1/1 [==============================] - 0s 20ms/step
0.05699634552001953
1/1 [==============================] - 0s 28ms/step
0.07399940490722656
1/1 [==============================] - 0s 22ms/step
0.06200122833251953
1/1 [==============================] - 0s 22ms/step
0.07300448417663574
1/1 [==============================] - 0s 28ms/step
0.07600212097167969
1/1 [==============================] - 0s 28ms/step
0.07200002670288086
1/1 [==============================] - 0s 20ms/step
0.06800055503845215
1/1 [==============================] - 0s 21ms/step
0.06398773193359375
1/1 [==============================] - 0s 25ms/step
0.07599973678588867
1/1 [==============================] - 0s 26ms/step
0.069000244140625
1/1 [==============================] - 0s 21ms/step
0.0650026798248291
1/1 [==============================] - 0s 22ms/step
0.06100630760192871
1/1 [==============================] - 0s 27ms/step
0.07000112533569336
1/1 [==============================] - 0s 21ms/step
0.06099987030029297
1/1 [==============================] - 0s 24ms/step
0.0670011043548584
1/1 [==============================] - 0s 28ms/step
0.07500100135803223
1/1 [==============================] - 0s 21ms/step
0.055002450942993164
1/1 [==============================] - 0s 22ms/step
0.0670008659362793
1/1 [==============================] - 0s 22ms/step
0.06199979782104492
1/1 [==============================] - 0s 20ms/step
0.056000709533691406
1/1 [==============================] - 0s 23ms/step
0.07000613212585449
1/1 [==============================] - 0s 25ms/step
0.06600117683410645
1/1 [==============================] - 0s 26ms/step
0.06500601768493652
1/1 [==============================] - 0s 21ms/step
0.061002492904663086
1/1 [==============================] - 0s 22ms/step
0.0650014877319336
1/1 [==============================] - 0s 21ms/step
0.06500101089477539
1/1 [==============================] - 0s 21ms/step
0.06300210952758789
1/1 [==============================] - 0s 34ms/step
0.07900071144104004
1/1 [==============================] - 0s 21ms/step
0.12599968910217285
1/1 [==============================] - 0s 26ms/step
0.07000017166137695
1/1 [==============================] - 0s 21ms/step
0.06599974632263184
1/1 [==============================] - 0s 22ms/step
0.058001041412353516
1/1 [==============================] - 0s 23ms/step
0.06699419021606445
1/1 [==============================] - 0s 23ms/step
0.06500530242919922
1/1 [==============================] - 0s 26ms/step
0.07100510597229004
1/1 [==============================] - 0s 20ms/step
0.06200218200683594
1/1 [==============================] - 0s 27ms/step
0.07101082801818848
1/1 [==============================] - 0s 26ms/step
0.06999921798706055
1/1 [==============================] - 0s 20ms/step
0.05799722671508789
1/1 [==============================] - 0s 22ms/step
0.06200385093688965
1/1 [==============================] - 0s 26ms/step
0.07200002670288086
1/1 [==============================] - 0s 25ms/step
0.06900215148925781
1/1 [==============================] - 0s 21ms/step
0.062003374099731445
1/1 [==============================] - 0s 29ms/step
0.07299995422363281
1/1 [==============================] - 0s 26ms/step
0.07500195503234863
1/1 [==============================] - 0s 23ms/step
0.07499980926513672
1/1 [==============================] - 0s 24ms/step
0.08999800682067871
1/1 [==============================] - 0s 24ms/step
0.06599926948547363
1/1 [==============================] - 0s 20ms/step
0.06400179862976074
1/1 [==============================] - 0s 20ms/step
0.06999874114990234
1/1 [==============================] - 0s 28ms/step
0.07700157165527344
1/1 [==============================] - 0s 19ms/step
0.06600117683410645
1/1 [==============================] - 0s 26ms/step
0.06600046157836914
1/1 [==============================] - 0s 27ms/step
0.07500004768371582
1/1 [==============================] - 0s 24ms/step
0.06302094459533691
1/1 [==============================] - 0s 23ms/step
0.06400179862976074
1/1 [==============================] - 0s 20ms/step
0.059000492095947266
1/1 [==============================] - 0s 28ms/step
0.08600950241088867
"""

# Calculate the average for the given input
average_result = calculate_average(input_data)
print(average_result) 