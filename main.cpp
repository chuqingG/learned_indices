#include "src/RecursiveModelIndex.h"
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
using namespace std;

int max_number = 1000;
int max_keylen = 32;
string dataset_name = "";

inline char *GetCmdArg(char **begin, char **end, const std::string &arg) {
    char **itr = std::find(begin, end, arg);
    if (itr != end && ++itr != end)
        return *itr;
    return nullptr;
}

int read_dataset_char(vector<char *> &values, string filename) {
    cout << "----- Processing file ----- " << filename << "max length"
         << max_keylen << endl;
    std::ifstream in(filename);
    int counter = 0;
    while (counter < max_number) {
        string str;
        std::getline(in, str);
        int len = str.size();
        int real_len = min(len, max_keylen);
        char *cptr = new char[real_len + 1];
        // strcpy(cptr, str.data());
        strncpy(cptr, str.data(), real_len);
        cptr[real_len] = '\0';

        if (real_len > 0) {
            values.push_back(cptr);
            if (real_len > max_keylen)
                max_keylen = real_len;
        }
        else {
            break;
        }
        counter++;
    }
    if (!counter) {
        perror("Cannot read dataset");
        exit(1);
    }
    cout << "----FINISH READ---- : " << counter
         << " items in total, max len = " << max_keylen << endl;
    in.close();
    return max_keylen;
}

int main(int argc, char *argv[]) {
    char *size_arg = GetCmdArg(argv, argv + argc, "-n");
    char *datasets_arg = GetCmdArg(argv, argv + argc, "-d");
    if (size_arg != nullptr) {
        const string size_str{size_arg};
        max_number = std::stoul(size_str);
    }

    if (datasets_arg != nullptr) {
        const string datasets_str{datasets_arg};
        dataset_name = datasets_str;
    }
    else {
        cout << "No dataset" << endl;
    }

    NetworkParameters firstStageParams;
    firstStageParams.batchSize = 256;
    firstStageParams.maxNumEpochs = 1000;
    firstStageParams.learningRate = 0.01;
    firstStageParams.numNeurons = 8;

    NetworkParameters secondStageParams;
    secondStageParams.batchSize = 64;
    secondStageParams.maxNumEpochs = 10000;
    secondStageParams.learningRate = 0.01;

    RecursiveModelIndex<uint64_t, int, 128> recursiveModelIndex(
        firstStageParams, secondStageParams, 256, 1e6);

    vector<char *> keys;
    read_dataset_char(keys, dataset_name);

    for (int i = 1; i < 1000; i++) {
        recursiveModelIndex.insert((uint64_t)(keys[i]), i);
    }

    recursiveModelIndex.train();

    for (int i = 999; i > 0; i--) {
        auto result = recursiveModelIndex.find((uint64_t)(keys[i]));
        std::cout << "Find key:" << (char *)((*result).first) << ", value is " << ((*result).second) << std::endl;
    }

    return 0;
}