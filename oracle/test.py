from tdc import Oracle
test_smi = "c1c2c(C3CCN(CC(O)C(O)(c4c(F)cc(F)cc4)Cn4ncnc4)CC3)c[nH]c2ccc1F"
oracle_GSK3B = Oracle(name = 'GSK3B')
oracle_JNK3 = Oracle(name = 'JNK3')
oracle_sa = Oracle(name = 'SA')
oracle_qed = Oracle(name = 'QED')
score0 = oracle_GSK3B(test_smi)
score1 = oracle_JNK3(test_smi)
score2 = (oracle_sa(test_smi) - 1) / 10
score3 = oracle_qed(test_smi)
print(score0)
print(score1)
print(score2)
print(score3)