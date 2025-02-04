const byte RED_LED = 12;    // Pin 12 -> PortB bit 4 (12-8)
const byte GREEN_LED = 11;  // Pin 11 -> PortB bit 3 (11-8)
const byte RED_LED_BIT = RED_LED - 8;
const byte GREEN_LED_BIT = GREEN_LED - 8;

void setup() {
  pinMode(RED_LED, OUTPUT);
  pinMode(GREEN_LED, OUTPUT);
  Serial.begin(4800);
}

void loop() {
  if (Serial.available()) {
    char c = Serial.read();
    if(c == '1'){
      PORTB |= _BV(RED_LED_BIT);
      PORTB &= ~_BV(GREEN_LED_BIT);
    }
    else if(c == '0'){
      PORTB &= ~_BV(RED_LED_BIT);
      PORTB |= _BV(GREEN_LED_BIT);
    }
  }
}
