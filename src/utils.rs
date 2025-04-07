// mostly rewrite by chatgpt4o
use candle_datasets::nlp::tinystories::Dataset;
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;
/// http://d2l.ai/_modules/d2l/torch.html#Vocab
#[derive(Debug, Clone)]
pub struct Vocab {
    idx_to_token: Vec<String>,
    token_to_idx: HashMap<String, usize>,
    token_freqs: Vec<(String, usize)>,
}

impl Vocab {
    pub fn new(tokens: &[String], min_freq: usize, reserved_tokens: Vec<String>) -> Self {
        let flat_tokens = tokens;

        let mut counter = HashMap::new();
        for token in flat_tokens {
            *counter.entry(token.clone()).or_insert(0) += 1;
        }
        // Count token frequencies
        let mut token_freqs: Vec<_> = counter.into_iter().collect();
        token_freqs.sort_by(|a, b| b.1.cmp(&a.1));
        // The list of unique tokens

        let unique_tokens = token_freqs
            .iter()
            .filter(|(_token, freq)| *freq >= min_freq)
            .map(|(token, _freq)| token.clone())
            .chain(reserved_tokens)
            .chain(once("<unk>".to_string()))
            .collect::<HashSet<_>>();
        let mut idx_to_token = unique_tokens.into_iter().collect::<Vec<_>>();
        idx_to_token.sort_unstable();

        let token_to_idx = idx_to_token
            .iter()
            .enumerate()
            .map(|(idx, token)| (token.clone(), idx))
            .collect();

        Vocab {
            idx_to_token,
            token_to_idx,
            token_freqs,
        }
    }

    pub fn len(&self) -> usize {
        self.idx_to_token.len()
    }

    pub fn get(&self, token: &str) -> usize {
        *self.token_to_idx.get(token).unwrap_or(&self.unk())
    }

    pub fn get_s(&self, tokens: &[String]) -> Vec<usize> {
        tokens.iter().map(|token| self.get(token)).collect()
    }

    pub fn to_tokens(&self, indices: &[usize]) -> Vec<String> {
        indices.iter().map(|&index| self.to_token(index)).collect()
    }
    pub fn to_token(&self, indice: usize) -> String {
        self.idx_to_token[indice].clone()
    }

    pub fn unk(&self) -> usize {
        *self.token_to_idx.get("<unk>").unwrap()
    }
}

use std::fs::File;
use std::io::{self, Read};
use std::iter::once;
/// http://d2l.ai/_modules/d2l/torch.html#TimeMachine
pub struct TimeMachine {
    batch_size: usize,
    num_steps: usize,
    num_train: usize,
    num_val: usize,
    pub vocab: Vocab,
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
            vocab: Vocab::new(&[], 0, vec![]),
            corpus: Vec::new(),
            X: Vec::new(),
            Y: Vec::new(),
        };
        let raw_text = tm.download();
        let tokens = tm.tokenize(tm.preprocess(&raw_text));
        let t = tokens.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        let (corpus, vocab) = tm.build(&t, Some(tm.vocab.clone()));
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
        text.chars()
            .map(|c| {
                if c.is_ascii_alphabetic() {
                    c.to_ascii_lowercase()
                } else {
                    ' '
                }
            })
            .collect()
    }

    fn tokenize(&self, text: Vec<char>) -> Vec<char> {
        text
    }

    pub fn build(&self, tokens: &[String], vocab: Option<Vocab>) -> (Vec<usize>, Vocab) {
        let vocab = vocab.unwrap_or(Vocab::new(tokens, 0, vec![]));
        let corpus: Vec<_> = tokens.iter().map(|x| vocab.get(x)).collect();
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
        if train {
            (X_data, Y_data)
        } else {
            todo!()
        }
    }
}
#[cfg(test)]
mod test {
    use std::collections::HashMap;
    use std::collections::HashSet;
    use std::io;

    use super::TimeMachine;
    use super::Vocab;
    #[test]

    fn test_Vocab() {
        let v = Vocab::new(&["aa".into(), "bb".into(), "cc".into()], 0, vec![]);
        assert_eq!(v.idx_to_token, ["<unk>", "aa", "bb", "cc",]);
        assert_eq!(
            HashMap::from_iter(v.token_freqs.clone()),
            [("bb".into(), 1), ("aa".into(), 1), ("cc".into(), 1)].into()
        );
        assert_eq!(
            HashMap::from_iter(v.token_to_idx.clone()),
            [
                ("bb".into(), 2),
                ("aa".into(), 1),
                ("cc".into(), 3),
                ("<unk>".into(), 0)
            ]
            .into()
        );
        assert_eq!(v.get_s(&["aa".into(), "bb".into()]), [1, 2]);
        assert_eq!(v.get("aa"), 1);
    }

    // #[test]
    // fn aaa() -> io::Result<()> {
    //     let tm = TimeMachine::new(32, 35, 10000, 5000)?;
    //     let (train_X, train_Y) = tm.get_dataloader(true);
    //     let (val_X, val_Y) = tm.get_dataloader(false);
    //     assert_ne!(
    //         train_X[0],
    //         [
    //             0, 1, 2, 3, 0, 4, 5, 2, 3, 5, 6, 7, 1, 4, 8, 2, 3, 3, 9, 10, 3, 1, 3, 3, 11, 3, 3,
    //             12, 2, 13, 13, 14, 3, 3, 3
    //         ]
    //     );
    //     assert_eq!(
    //         train_Y[0],
    //         [
    //             1, 2, 3, 0, 4, 5, 2, 3, 5, 6, 7, 1, 4, 8, 2, 3, 3, 9, 10, 3, 1, 3, 3, 11, 3, 3, 12,
    //             2, 13, 13, 14, 3, 3, 3, 3
    //         ]
    //     );
    //     assert_eq!(
    //         val_X[0],
    //         [
    //             12, 6, 14, 3, 4, 16, 18, 15, 10, 3, 4, 8, 3, 4, 0, 3, 3, 6, 8, 23, 3, 14, 18, 5, 2,
    //             3, 0, 15, 6, 8, 14, 19, 6, 15, 2
    //         ]
    //     );
    //     assert_eq!(
    //         val_Y[0],
    //         [
    //             6, 14, 3, 4, 16, 18, 15, 10, 3, 4, 8, 3, 4, 0, 3, 3, 6, 8, 23, 3, 14, 18, 5, 2, 3,
    //             0, 15, 6, 8, 14, 19, 6, 15, 2, 8
    //         ]
    //     );

    //     Ok(())
    // }
}
