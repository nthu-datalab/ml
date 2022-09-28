bool sameString(string s1, string s2, int i) {
    for(int j = 0; j < s2.size(); j++) {
        if(s1[i + j] != s2[j]) return false;
    }
    return true;
}
int str2str(string s1, string s2) {
    if(s1.size() < s2.size()) 
        return -1;
    for(int i = 0; i <= s1.size() - s2.size(); i++) {
        if(sameString(s1, s2, i))
            return i;
    }
    return -1'
}