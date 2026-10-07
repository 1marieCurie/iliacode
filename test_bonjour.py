from bonjour import greeting

def test_greeting_khadija():
    assert greeting("Khadija") == "Bonjour Khadija" 

def test_greeting_vide(): 
    assert greeting("") == "Bonjour "