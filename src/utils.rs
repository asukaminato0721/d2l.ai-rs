// mostly rewrite by chatgpt4o
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;

/// http://d2l.ai/_modules/d2l/torch.html#Vocab
#[derive(Debug)]
pub struct Vocab {
    idx_to_token: Vec<String>,
    token_to_idx: HashMap<String, usize>,
    token_freqs: Vec<(String, usize)>,
}

impl Vocab {
    pub fn new(tokens: Vec<Vec<String>>, min_freq: usize, reserved_tokens: Vec<String>) -> Self {
        let mut flat_tokens = Vec::new();
        for line in tokens {
            for token in line {
                flat_tokens.push(token);
            }
        }

        let mut counter = HashMap::new();
        for token in flat_tokens {
            *counter.entry(token).or_insert(0) += 1;
        }

        let mut token_freqs: Vec<_> = counter.into_iter().collect();
        token_freqs.sort_by(|a, b| b.1.cmp(&a.1));

        let mut unique_tokens: HashSet<String> = HashSet::new();
        unique_tokens.insert("<unk>".to_string());
        for token in reserved_tokens {
            unique_tokens.insert(token);
        }

        for (token, freq) in &token_freqs {
            if *freq >= min_freq {
                unique_tokens.insert(token.clone());
            }
        }

        let idx_to_token: Vec<String> = unique_tokens.into_iter().collect();
        let mut token_to_idx = HashMap::new();
        for (idx, token) in idx_to_token.iter().enumerate() {
            token_to_idx.insert(token.clone(), idx);
        }

        Vocab {
            idx_to_token,
            token_to_idx,
            token_freqs,
        }
    }

    pub fn len(&self) -> usize {
        self.idx_to_token.len()
    }

    pub fn get(&self, tokens: &Vec<String>) -> Vec<usize> {
        tokens
            .iter()
            .map(|token| *self.token_to_idx.get(token).unwrap_or(&self.unk()))
            .collect()
    }

    pub fn to_tokens(&self, indices: Vec<usize>) -> Vec<String> {
        indices
            .iter()
            .map(|&index| self.idx_to_token[index].clone())
            .collect()
    }

    pub fn unk(&self) -> usize {
        *self.token_to_idx.get("<unk>").unwrap()
    }
}

use std::fs::File;
use std::io::{self, Read};
/// http://d2l.ai/_modules/d2l/torch.html#TimeMachine
pub struct TimeMachine {
    batch_size: usize,
    num_steps: usize,
    num_train: usize,
    num_val: usize,
    pub vocab: HashMap<char, usize>,
    corpus: Vec<usize>,
    X: Vec<Vec<usize>>,
    Y: Vec<Vec<usize>>,
}

impl TimeMachine {
    pub fn new(
        batch_size: usize,
        num_steps: usize,
        num_train: usize,
        num_val: usize,
    ) -> io::Result<Self> {
        let mut tm = TimeMachine {
            batch_size,
            num_steps,
            num_train,
            num_val,
            vocab: HashMap::new(),
            corpus: Vec::new(),
            X: Vec::new(),
            Y: Vec::new(),
        };
        let raw_text = tm.download();
        let tokens = tm.tokenize(tm.preprocess(&raw_text));
        let (corpus, vocab) = tm.build(tokens);
        tm.vocab = vocab;
        tm.corpus = corpus.clone();
        tm.prepare_data(corpus);
        Ok(tm)
    }

    fn download(&self) -> String {
        use std::fs;
        fs::read_to_string("data/timemachine.txt").expect("Should have been able to read the file")
    }

    fn preprocess(&self, text: &str) -> Vec<char> {
        let mut result = vec![];
        for c in text.chars() {
            if c.is_ascii_alphabetic() {
                result.push(c.to_ascii_lowercase());
            } else {
                result.push(' ');
            }
        }
        result
    }

    fn tokenize(&self, text: Vec<char>) -> Vec<char> {
        text
    }

    pub fn build(&self, tokens: Vec<char>) -> (Vec<usize>, HashMap<char, usize>) {
        let mut vocab: HashMap<char, usize> = HashMap::new();
        let mut corpus = Vec::new();

        for token in tokens {
            let count = vocab.len();
            let entry = vocab.entry(token).or_insert(count);
            corpus.push(*entry);
        }

        (corpus, vocab)
    }

    fn prepare_data(&mut self, corpus: Vec<usize>) {
        let array: Vec<Vec<usize>> = (0..corpus.len() - self.num_steps)
            .map(|i| corpus[i..i + self.num_steps + 1].to_vec())
            .collect();

        self.X = array
            .iter()
            .map(|seq| seq[..self.num_steps].to_vec())
            .collect();
        self.Y = array.iter().map(|seq| seq[1..].to_vec()).collect();
    }

    pub fn get_dataloader(&self, train: bool) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
        let idx_range = if train {
            0..self.num_train
        } else {
            self.num_train..self.num_train + self.num_val
        };

        let X_data = self.X[idx_range.clone()].to_vec();
        let Y_data = self.Y[idx_range].to_vec();

        (X_data, Y_data)
    }
}
#[cfg(test)]
mod test {
    use std::io;

    use super::TimeMachine;

    #[test]
    fn aaa() -> io::Result<()> {
        let tm = TimeMachine::new(32, 35, 10000, 5000)?;
        let (train_X, train_Y) = tm.get_dataloader(true);
        let (val_X, val_Y) = tm.get_dataloader(false);

        // Print some data to verify
        println!("Train X: {:?}", &train_X[0..1]);
        println!("Train Y: {:?}", &train_Y[0..1]);
        println!("Validation X: {:?}", &val_X[0..1]);
        println!("Validation Y: {:?}", &val_Y[0..1]);

        Ok(())
    }
}
