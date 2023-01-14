# QrCode stegnography

<br>


## Encoding
```
clear && py qrcode_stegnography2.py -e mandrill.png -t "vedat önal merhaba"
```

<br>

```
clear && py qrcode_stegnography2.py -e mandrill.png -t "vedat önal merhaba" -q
```

<br>

## Decoding
```
clear && py qrcode_stegnography2.py -d mandrill_encoded.png
```

<br>

```
clear && py qrcode_stegnography2.py -d mandrill_encoded.png -q
```

SONUÇLAR
|   | MSE | RMSE | PSNR | SSIM |
|---|---|---|---|---|
| LENA | 0,00469 | 0,06848 | 71,4194 | 0,99999 |
| LENA RGB | 0,00912 | 0,09550 | 68,5303  | 0,99998 |
| MANDRİLL | 0,01318 | 0,11480 | 66,9321 | 1,00000 | 
| MANDRİLL RGB | 0,00365 | 0,06039 | 72,5120 | 1,00000 | 
| PEPPERS RGB | 0,00015 | 0,01224 | 86,3747| 1,00000 | 

Resimler için: [Tıkla](https://sipi.usc.edu/database/database.php?volume=misc)
